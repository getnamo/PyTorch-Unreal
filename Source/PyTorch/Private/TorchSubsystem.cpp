#include "TorchSubsystem.h"
#include "PyTorch.h"
#include "PythonCLISettings.h"
#include "PythonEnvironment.h"
#include "PythonJob.h"
#include "PythonService.h"
#include "PythonSubsystem.h"
#include "TorchModel.h"
#include "TorchNNE.h"
#include "TorchOutputComponent.h"
#include "TorchSettings.h"
#include "NNEModelData.h"
#include "AssetRegistry/AssetRegistryModule.h"
#include "AssetRegistry/IAssetRegistry.h"
#include "Dom/JsonObject.h"
#include "Engine/Engine.h"
#include "HAL/FileManager.h"
#include "Misc/CoreDelegates.h"
#include "Misc/Paths.h"
#include "Modules/ModuleManager.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#if WITH_EDITOR
#include "Misc/FileHelper.h"
#include "Misc/PackageName.h"
#include "UObject/Package.h"
#include "UObject/SavePackage.h"
#endif

namespace TorchSubsystemPrivate
{
	static TArray<int64> RowShape(const FPyTensor& T, bool bBatched)
	{
		TArray<int64> Shape = T.Shape;
		if (bBatched && Shape.Num() > 0)
		{
			Shape.RemoveAt(0);
		}
		return Shape;
	}

	static FString ShapeString(const TArray<int64>& Shape)
	{
		FString S;
		for (int64 D : Shape)
		{
			S += FString::Printf(TEXT("%lld,"), D);
		}
		return S;
	}

	/** Requests stack together when they have the same inputs (names, dtypes, per-row shapes), JSON values and layouts. */
	static FString GroupKey(const TMap<FString, FPyTensor>& Inputs, const FString& JsonInputs, const FTorchRunOptions& Options)
	{
		TArray<FString> Names;
		Inputs.GetKeys(Names);
		Names.Sort();
		FString Key = JsonInputs + TEXT("#");
		for (const FString& Name : Names)
		{
			const FPyTensor& T = Inputs[Name];
			Key += Name + TEXT(":") + FPyTensor::DTypeName(T.DType) + TEXT(":") + ShapeString(RowShape(T, Options.bBatched));
			if (const FString* Layout = Options.Layouts.Find(Name))
			{
				Key += TEXT(":") + *Layout;
			}
			Key += TEXT(";");
		}
		return Key;
	}

	static FPyTensor SliceRows(const FPyTensor& T, int64 Start, int64 Count, bool bKeepBatchDim)
	{
		if (T.Shape.Num() == 0)
		{
			return T;
		}
		int64 RowElements = 1;
		for (int32 i = 1; i < T.Shape.Num(); ++i)
		{
			RowElements *= T.Shape[i];
		}
		const int64 RowBytes = RowElements * FPyTensor::ItemSize(T.DType);
		FPyTensor Out;
		Out.DType = T.DType;
		Out.Shape = T.Shape;
		Out.Shape[0] = Count;
		if (Start * RowBytes + Count * RowBytes <= T.Data.Num())
		{
			Out.Data.Append(T.Data.GetData() + Start * RowBytes, Count * RowBytes);
		}
		if (!bKeepBatchDim && Count == 1)
		{
			Out.Shape.RemoveAt(0);
		}
		return Out;
	}
}

UTorchSubsystem* UTorchSubsystem::Get()
{
	return GEngine ? GEngine->GetEngineSubsystem<UTorchSubsystem>() : nullptr;
}

void UTorchSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	Collection.InitializeDependency<UPythonSubsystem>();
	Super::Initialize(Collection);
	EndFrameHandle = FCoreDelegates::OnEndFrame.AddUObject(this, &UTorchSubsystem::HandleEndFrame);
	if (!IsRunningCommandlet())
	{
		PostEngineInitHandle = FCoreDelegates::GetOnPostEngineInit().AddUObject(this, &UTorchSubsystem::LoadStartupModels);
	}
}

void UTorchSubsystem::Deinitialize()
{
	FCoreDelegates::OnEndFrame.Remove(EndFrameHandle);
	FCoreDelegates::GetOnPostEngineInit().Remove(PostEngineInitHandle);
	for (TPair<FString, TArray<FRequest>>& Pair : Pending)
	{
		Fail(Pair.Value, TEXT("the engine is shutting down"));
	}
	Pending.Reset();
	Super::Deinitialize();
}

FString UTorchSubsystem::ModelKey(const UTorchModel* Model)
{
	return Model ? FString::Printf(TEXT("%s|%s|%d"), *Model->Script, *Model->GetModelName(), (int32)Model->Process) : FString();
}

UPythonService* UTorchSubsystem::GetService(const UTorchModel* Model)
{
	UPythonSubsystem* Python = UPythonSubsystem::Get();
	if (!Python || !Model || Model->Script.IsEmpty())
	{
		return nullptr;
	}
	//The models of one script share its service; a dedicated model gets a process of its own
	const FString Name = Model->Process == EPythonServiceProcess::Dedicated
		? FString::Printf(TEXT("torch:%s#%s"), *Model->Script, *Model->GetModelName())
		: TEXT("torch:") + Model->Script;
	UPythonService* Service = Python->GetService(Name, Model->Script, Model->Process, {});
	BindService(Service);
	return Service;
}

void UTorchSubsystem::BindService(UPythonService* Service)
{
	if (!Service || BoundServices.Contains(Service))
	{
		return;
	}
	BoundServices.Add(Service);
	Service->OnScriptStopped.AddDynamic(this, &UTorchSubsystem::HandleServiceStopped);
	Service->OnScriptError.AddDynamic(this, &UTorchSubsystem::HandleServiceError);
	Service->OnNativeEvent(TEXT("_torch_stream_result")).AddUObject(this, &UTorchSubsystem::HandleStreamResult);
}

void UTorchSubsystem::HandleServiceStopped(const FString&)
{
	ResetStoppedModels();
}

void UTorchSubsystem::HandleServiceError(const FString&, const FString&)
{
	ResetStoppedModels();
}

void UTorchSubsystem::ResetStoppedModels()
{
	//A script that stopped (not reloading) or a process that died took its models with it: load again on next use
	UPythonSubsystem* Python = UPythonSubsystem::Get();
	for (TPair<FString, FModelEntry>& Pair : Models)
	{
		TArray<FString> Parts;
		Pair.Key.ParseIntoArray(Parts, TEXT("|"), false);
		if (Parts.Num() < 3)
		{
			continue;
		}
		const bool bDedicated = Parts[2] == FString::FromInt((int32)EPythonServiceProcess::Dedicated);
		const FString Name = bDedicated ? FString::Printf(TEXT("torch:%s#%s"), *Parts[0], *Parts[1]) : TEXT("torch:") + Parts[0];
		UPythonService* Service = Python ? Python->FindService(Name) : nullptr;
		if (!Service || (!Service->IsRunning() && !Service->IsStarting()))
		{
			Pair.Value.State = ETorchModelState::NotLoaded;
		}
	}
}

ETorchModelState UTorchSubsystem::GetModelState(const UTorchModel* Model) const
{
	if (ModelUsesNNE(Model))
	{
		return ETorchModelState::Ready; //NNE instances are made on first use (or Load Model)
	}
	const FModelEntry* Entry = Models.Find(ModelKey(Model));
	return Entry ? Entry->State : ETorchModelState::NotLoaded;
}

FString UTorchSubsystem::GetModelDevice(const UTorchModel* Model) const
{
	const FModelEntry* Entry = Models.Find(ModelKey(Model));
	return Entry ? Entry->Device : FString();
}

void UTorchSubsystem::EnsureLoadSent(UTorchModel* Model)
{
	FModelEntry& Entry = Models.FindOrAdd(ModelKey(Model));
	if (Entry.State != ETorchModelState::NotLoaded)
	{
		return;
	}
	UPythonService* Service = GetService(Model);
	if (!Service)
	{
		Entry.State = ETorchModelState::Failed;
		Entry.Error = TEXT("Python is not available (or the model has no Script)");
		return;
	}
	Entry.State = ETorchModelState::Loading;
	const FString Key = ModelKey(Model);
	const FString Args = PythonWire::ToJsonString(TArray<TSharedPtr<FJsonValue>>{
		MakeShared<FJsonValueString>(Model->GetModelName()), MakeShared<FJsonValueObject>(Model->MakeLoadOptions()) });
	TWeakObjectPtr<UTorchSubsystem> WeakSelf(this);
	TWeakObjectPtr<UTorchModel> WeakModel(Model);
	Service->CallFunction(TEXT("_torch_load"), Args, {}, [WeakSelf, WeakModel, Key](const FPythonCallResult& R)
	{
		UTorchSubsystem* Self = WeakSelf.Get();
		if (!Self)
		{
			return;
		}
		FModelEntry& E = Self->Models.FindOrAdd(Key);
		E.State = R.bSuccess ? ETorchModelState::Ready : ETorchModelState::Failed;
		E.Error = R.Error;
		const TSharedPtr<FJsonObject>* Obj = nullptr;
		if (R.bSuccess && R.Result.IsValid() && R.Result->TryGetObject(Obj))
		{
			(*Obj)->TryGetStringField(TEXT("device"), E.Device);
			(*Obj)->TryGetNumberField(TEXT("load_ms"), E.LoadMs);
			UE_LOG(LogTorch, Log, TEXT("Model '%s' loaded on %s in %.0f ms"), *Key, *E.Device, E.LoadMs);
		}
		else if (!R.bSuccess)
		{
			UE_LOG(LogTorch, Error, TEXT("Model '%s' failed to load: %s"), *Key, *R.Error);
		}
		TArray<TFunction<void(bool, const FString&)>> Waiters = MoveTemp(E.Waiters);
		E.Waiters.Reset();
		for (TFunction<void(bool, const FString&)>& W : Waiters)
		{
			W(R.bSuccess, R.Error);
		}
		Self->OnModelLoaded.Broadcast(WeakModel.Get(), R.bSuccess, R.Error);
	});
}

void UTorchSubsystem::LoadModel(UTorchModel* Model, TFunction<void(bool, const FString&)> OnDone)
{
	if (!Model)
	{
		if (OnDone)
		{
			OnDone(false, TEXT("no model"));
		}
		return;
	}
	if (ModelUsesNNE(Model))
	{
		FString Error;
		bool bOk = true;
		for (const FTorchNNEMethod& M : Model->NNEMethods)
		{
			bOk = bOk && FTorchNNE::Get().Prepare(Model, M.Method, Error);
		}
		OnModelLoaded.Broadcast(Model, bOk, Error);
		if (OnDone)
		{
			OnDone(bOk, Error);
		}
		return;
	}
	FModelEntry& Entry = Models.FindOrAdd(ModelKey(Model));
	if (Entry.State == ETorchModelState::Ready)
	{
		if (OnDone)
		{
			OnDone(true, FString());
		}
		return;
	}
	if (Entry.State == ETorchModelState::Failed)
	{
		Entry.State = ETorchModelState::NotLoaded; //Asked explicitly: try again
	}
	if (OnDone)
	{
		Entry.Waiters.Add(MoveTemp(OnDone));
	}
	EnsureLoadSent(Model);
	FModelEntry& After = Models.FindOrAdd(ModelKey(Model));
	if (After.State == ETorchModelState::Failed && After.Waiters.Num() > 0)
	{
		TArray<TFunction<void(bool, const FString&)>> Waiters = MoveTemp(After.Waiters);
		After.Waiters.Reset();
		for (TFunction<void(bool, const FString&)>& W : Waiters)
		{
			W(false, After.Error);
		}
	}
}

void UTorchSubsystem::UnloadModel(UTorchModel* Model)
{
	FTorchNNE::Get().Reset(Model);
	if (ModelUsesNNE(Model) || !IsPythonAvailable())
	{
		return;
	}
	UPythonService* Service = GetService(Model);
	if (!Service)
	{
		return;
	}
	Service->CallFunction(TEXT("_torch_unload"), PythonWire::ToJsonString(TArray<TSharedPtr<FJsonValue>>{ MakeShared<FJsonValueString>(Model->GetModelName()) }), {}, nullptr);
	Models.FindOrAdd(ModelKey(Model)).State = ETorchModelState::NotLoaded;
}

void UTorchSubsystem::ReloadScript(UTorchModel* Model)
{
	if (UPythonService* Service = GetService(Model))
	{
		Service->Reload();
	}
}

void UTorchSubsystem::FetchSignature(UTorchModel* Model, TFunction<void(bool, const FString&)> OnDone)
{
	UPythonService* Service = GetService(Model);
	if (!Service)
	{
		OnDone(false, TEXT("Python is not available (or the model has no Script)"));
		return;
	}
	TWeakObjectPtr<UTorchModel> WeakModel(Model);
	Service->CallFunction(TEXT("_torch_list"), TEXT("[]"), {}, [WeakModel, OnDone = MoveTemp(OnDone)](const FPythonCallResult& R)
	{
		UTorchModel* M = WeakModel.Get();
		const TSharedPtr<FJsonObject>* Obj = nullptr;
		if (!M || !R.bSuccess || !R.Result.IsValid() || !R.Result->TryGetObject(Obj))
		{
			OnDone(false, R.bSuccess ? TEXT("no result") : R.Error);
			return;
		}
		const TArray<TSharedPtr<FJsonValue>>* ModelList = nullptr;
		const TArray<TSharedPtr<FJsonValue>>* JobList = nullptr;
		(*Obj)->TryGetArrayField(TEXT("jobs"), JobList);
		if ((*Obj)->TryGetArrayField(TEXT("models"), ModelList))
		{
			for (const TSharedPtr<FJsonValue>& V : *ModelList)
			{
				const TSharedPtr<FJsonObject> Sig = V->AsObject();
				if (Sig.IsValid() && Sig->GetStringField(TEXT("model")) == M->GetModelName())
				{
					M->ApplySignature(Sig, JobList);
					OnDone(true, FString());
					return;
				}
			}
		}
		OnDone(false, FString::Printf(TEXT("the script has no @ut.model('%s')"), *M->GetModelName()));
	});
}

void UTorchSubsystem::GetStats(UTorchModel* Model, TFunction<void(const FPythonCallResult&)> OnResult, bool bReset)
{
	UPythonService* Service = GetService(Model);
	if (!Service)
	{
		FPythonCallResult R;
		R.Error = TEXT("Python is not available");
		OnResult(R);
		return;
	}
	Service->CallFunction(TEXT("_torch_stats"), PythonWire::ToJsonString(TArray<TSharedPtr<FJsonValue>>{
		MakeShared<FJsonValueString>(Model->GetModelName()), MakeShared<FJsonValueBoolean>(bReset) }), {}, MoveTemp(OnResult));
}

void UTorchSubsystem::Run(UTorchModel* Model, const FString& Method, const TMap<FString, FPyTensor>& Inputs, const FString& JsonInputs,
	TFunction<void(const FTorchResult&)> OnResult, const FTorchRunOptions& Options)
{
	FRequest Request;
	Request.Model = Model;
	Request.Method = Method;
	Request.Inputs = Inputs;
	Request.JsonInputs = JsonInputs.TrimStartAndEnd();
	Request.Options = Options;
	Request.OnResult = MoveTemp(OnResult);
	Request.Start = FPlatformTime::Seconds();
	const bool bNNE = UsesNNE(Model, Method);
	if (!Model || (Model->Script.IsEmpty() && !bNNE))
	{
		TArray<FRequest> One;
		One.Add(MoveTemp(Request));
		Fail(One, Model ? TEXT("the Torch Model has no Script") : TEXT("no Torch Model"));
		return;
	}
	for (const TPair<FString, FPyTensor>& Pair : Inputs)
	{
		if (!Pair.Value.IsValid())
		{
			TArray<FRequest> One;
			One.Add(MoveTemp(Request));
			Fail(One, FString::Printf(TEXT("input '%s': the tensor's data doesn't match its dtype and shape"), *Pair.Key));
			return;
		}
		if (Options.bBatched)
		{
			Request.Rows = Pair.Value.Shape.Num() > 0 ? (int32)Pair.Value.Shape[0] : 1;
		}
	}
	if (!bNNE)
	{
		EnsureLoadSent(Model); //Calls to the script run in order: the load is first
	}
	if (Options.bImmediate || !Model->bStackRequests)
	{
		TArray<FRequest> One;
		One.Add(MoveTemp(Request));
		SendGroup(Model, Method, MoveTemp(One));
		return;
	}
	const FString Key = ModelKey(Model) + TEXT("|") + Method + TEXT("|") + TorchSubsystemPrivate::GroupKey(Inputs, Request.JsonInputs, Options);
	Pending.FindOrAdd(Key).Add(MoveTemp(Request));
}

void UTorchSubsystem::HandleEndFrame()
{
	if (Pending.Num() > 0)
	{
		Flush();
	}
}

void UTorchSubsystem::Flush()
{
	TMap<FString, TArray<FRequest>> Groups = MoveTemp(Pending);
	Pending.Reset();
	for (TPair<FString, TArray<FRequest>>& Pair : Groups)
	{
		TArray<FRequest>& Requests = Pair.Value;
		UTorchModel* Model = Requests.Num() ? Requests[0].Model.Get() : nullptr;
		if (!Model)
		{
			Fail(Requests, TEXT("the Torch Model was destroyed"));
			continue;
		}
		const FString Method = Requests[0].Method;
		const int32 Max = FMath::Max(1, Model->MaxStackedRequests);
		for (int32 Start = 0; Start < Requests.Num(); Start += Max)
		{
			TArray<FRequest> Chunk;
			for (int32 i = Start; i < FMath::Min(Start + Max, Requests.Num()); ++i)
			{
				Chunk.Add(MoveTemp(Requests[i]));
			}
			SendGroup(Model, Method, MoveTemp(Chunk));
		}
	}
}

void UTorchSubsystem::Fail(TArray<FRequest>& Requests, const FString& Error)
{
	for (FRequest& R : Requests)
	{
		if (R.OnResult)
		{
			FTorchResult Out;
			Out.Error = Error;
			Out.LatencyMs = (float)((FPlatformTime::Seconds() - R.Start) * 1000.0);
			R.OnResult(Out);
		}
	}
}

void UTorchSubsystem::StackInputs(const TArray<FRequest>& Requests, TArray<FString>& OutNames, TArray<FPyTensor>& OutTensors, int64& OutRows)
{
	using namespace TorchSubsystemPrivate;
	const FRequest& First = Requests[0];
	OutNames.Reset();
	OutTensors.Reset();
	First.Inputs.GetKeys(OutNames);
	OutNames.Sort();
	OutRows = 0;
	for (const FRequest& R : Requests)
	{
		OutRows += R.Rows;
	}
	for (const FString& Name : OutNames)
	{
		const FPyTensor& T0 = First.Inputs[Name];
		FPyTensor& Stacked = OutTensors.AddDefaulted_GetRef();
		Stacked.DType = T0.DType;
		for (const FRequest& R : Requests)
		{
			Stacked.Data.Append(R.Inputs[Name].Data);
		}
		Stacked.Shape = { OutRows };
		Stacked.Shape.Append(RowShape(T0, First.Options.bBatched));
	}
}

bool UTorchSubsystem::IsPythonAvailable()
{
	if (FPythonEnvironment::IsPackagedBuild() && GetDefault<UPythonCLISettings>()->PackagedEnvMode == EPythonPackagedEnvMode::None)
	{
		return false;
	}
	return FPythonEnvironment::HasBundledRuntime();
}

bool UTorchSubsystem::UsesNNE(const UTorchModel* Model, const FString& Method) const
{
	if (!Model || !Model->FindNNEMethod(Method))
	{
		return false;
	}
	return Model->Backend == ETorchBackend::NNE || (Model->Backend == ETorchBackend::Auto && !IsPythonAvailable());
}

bool UTorchSubsystem::ModelUsesNNE(const UTorchModel* Model) const
{
	return Model && Model->NNEMethods.Num() > 0 && (Model->Backend == ETorchBackend::NNE || (Model->Backend == ETorchBackend::Auto && !IsPythonAvailable()));
}

void UTorchSubsystem::SendGroupNNE(UTorchModel* Model, const FString& Method, TArray<FRequest>&& Requests)
{
	TArray<FString> Names;
	TArray<FPyTensor> Tensors;
	int64 Total = 0;
	StackInputs(Requests, Names, Tensors, Total);
	TMap<FString, FPyTensor> Inputs;
	for (int32 i = 0; i < Names.Num(); ++i)
	{
		Inputs.Add(Names[i], MoveTemp(Tensors[i]));
	}
	const TMap<FString, FString> Layouts = Requests[0].Options.Layouts;
	TSharedRef<TArray<FRequest>> Shared = MakeShared<TArray<FRequest>>(MoveTemp(Requests));
	FTorchNNE::Get().Run(Model, Method, MoveTemp(Inputs), Total, Layouts, [Shared, Total](FTorchNNE::FResult&& R)
	{
		TArray<FRequest>& Requests = *Shared;
		const double Now = FPlatformTime::Seconds();
		if (!R.bSuccess)
		{
			for (FRequest& Req : Requests)
			{
				if (Req.OnResult)
				{
					FTorchResult Out;
					Out.Error = R.Error;
					Out.Backend = TEXT("nne");
					Out.LatencyMs = (float)((Now - Req.Start) * 1000.0);
					Req.OnResult(Out);
				}
			}
			return;
		}
		//The same reply shape as Python's, so the results split per request the same way
		TSharedRef<FJsonObject> Reply = MakeShared<FJsonObject>();
		TSharedRef<FJsonObject> Outputs = MakeShared<FJsonObject>();
		TArray<TSharedPtr<FJsonValue>> PerRow;
		for (int32 i = 0; i < R.Names.Num(); ++i)
		{
			TSharedRef<FJsonObject> Desc = MakeShared<FJsonObject>();
			Desc->SetNumberField(TEXT("_tensor"), i);
			Outputs->SetObjectField(R.Names[i], Desc);
			if (R.Outputs[i].Shape.Num() > 0 && R.Outputs[i].Shape[0] == Total)
			{
				PerRow.Add(MakeShared<FJsonValueString>(R.Names[i]));
			}
		}
		Reply->SetObjectField(TEXT("outputs"), Outputs);
		Reply->SetArrayField(TEXT("per_row"), PerRow);
		Reply->SetNumberField(TEXT("batch"), (double)Total);
		Reply->SetNumberField(TEXT("ms"), R.Ms);
		int64 Offset = 0;
		for (FRequest& Req : Requests)
		{
			FTorchResult Out = MakeResult(Reply, R.Outputs, Offset, Req.Rows, Req.Options.bBatched);
			Out.Backend = TEXT("nne:") + R.Runtime;
			Out.LatencyMs = (float)((Now - Req.Start) * 1000.0);
			Offset += Req.Rows;
			if (Req.OnResult)
			{
				Req.OnResult(Out);
			}
		}
	});
}

void UTorchSubsystem::SendGroup(UTorchModel* Model, const FString& Method, TArray<FRequest>&& Requests)
{
	using namespace TorchSubsystemPrivate;
	if (UsesNNE(Model, Method))
	{
		SendGroupNNE(Model, Method, MoveTemp(Requests));
		return;
	}
	UPythonService* Service = IsPythonAvailable() ? GetService(Model) : nullptr;
	if (!Service)
	{
		Fail(Requests, IsPythonAvailable() ? TEXT("Python is not available")
			: FString::Printf(TEXT("this build has no Python, and %s has no NNE export of '%s' (Export To NNE on the Torch Model)"), *Model->GetName(), *Method));
		return;
	}

	//One call: each tensor input stacked on dim 0 (one entry of `rows` per request), the JSON inputs shared
	const FRequest& First = Requests[0];
	TSharedRef<FJsonObject> InputsObj = MakeShared<FJsonObject>();
	if (!First.JsonInputs.IsEmpty())
	{
		TSharedPtr<FJsonObject> Json;
		if (!FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(First.JsonInputs), Json) || !Json.IsValid())
		{
			Fail(Requests, FString::Printf(TEXT("JSON inputs must be a JSON object: %s"), *First.JsonInputs));
			return;
		}
		InputsObj->Values = Json->Values;
	}
	TArray<FString> Names;
	TArray<FPyTensor> Tensors;
	int64 TotalRows = 0;
	StackInputs(Requests, Names, Tensors, TotalRows);
	for (int32 i = 0; i < Names.Num(); ++i)
	{
		TSharedRef<FJsonObject> Desc = MakeShared<FJsonObject>();
		Desc->SetNumberField(TEXT("_tensor"), i);
		Desc->SetStringField(TEXT("dtype"), FPyTensor::DTypeName(Tensors[i].DType));
		TArray<TSharedPtr<FJsonValue>> ShapeValues;
		for (int64 D : Tensors[i].Shape)
		{
			ShapeValues.Add(MakeShared<FJsonValueNumber>((double)D));
		}
		Desc->SetArrayField(TEXT("shape"), ShapeValues);
		InputsObj->SetObjectField(Names[i], Desc);
	}
	TArray<TSharedPtr<FJsonValue>> Rows;
	for (const FRequest& R : Requests)
	{
		Rows.Add(MakeShared<FJsonValueNumber>(R.Rows));
	}
	TSharedRef<FJsonObject> CallOptions = MakeShared<FJsonObject>();
	if (First.Options.Layouts.Num())
	{
		TSharedRef<FJsonObject> Layouts = MakeShared<FJsonObject>();
		for (const TPair<FString, FString>& L : First.Options.Layouts)
		{
			Layouts->SetStringField(L.Key, L.Value);
		}
		CallOptions->SetObjectField(TEXT("layouts"), Layouts);
	}
	const FString Args = PythonWire::ToJsonString(TArray<TSharedPtr<FJsonValue>>{
		MakeShared<FJsonValueString>(Model->GetModelName()), MakeShared<FJsonValueString>(Method), MakeShared<FJsonValueObject>(InputsObj),
		MakeShared<FJsonValueArray>(Rows), MakeShared<FJsonValueObject>(CallOptions) });

	TSharedRef<TArray<FRequest>> Shared = MakeShared<TArray<FRequest>>(MoveTemp(Requests));
	Service->CallFunction(TEXT("_torch_run"), Args, Tensors, [Shared](const FPythonCallResult& R)
	{
		TArray<FRequest>& Requests = *Shared;
		if (!R.bSuccess)
		{
			for (FRequest& Req : Requests)
			{
				if (Req.OnResult)
				{
					FTorchResult Out;
					Out.Error = R.Error;
					Out.Traceback = R.Traceback;
					Out.Backend = TEXT("python");
					Out.LatencyMs = (float)((FPlatformTime::Seconds() - Req.Start) * 1000.0);
					Req.OnResult(Out);
				}
			}
			return;
		}
		const TSharedPtr<FJsonObject>* Result = nullptr;
		if (!R.Result.IsValid() || !R.Result->TryGetObject(Result))
		{
			Fail(Requests, TEXT("malformed reply from unreal_torch"));
			return;
		}
		const double Now = FPlatformTime::Seconds();
		int64 Offset = 0;
		for (FRequest& Req : Requests)
		{
			FTorchResult Out = MakeResult(*Result, R.Tensors, Offset, Req.Rows, Req.Options.bBatched);
			Out.LatencyMs = (float)((Now - Req.Start) * 1000.0);
			Offset += Req.Rows;
			if (Req.OnResult)
			{
				Req.OnResult(Out);
			}
		}
	});
}

FTorchResult UTorchSubsystem::MakeResult(const TSharedPtr<FJsonObject>& Reply, const TArray<FPyTensor>& Tensors, int64 Offset, int32 Rows, bool bBatched)
{
	using namespace TorchSubsystemPrivate;
	FTorchResult Out;
	Out.Backend = TEXT("python");
	const TSharedPtr<FJsonObject>* Outputs = nullptr;
	if (!Reply.IsValid() || !Reply->TryGetObjectField(TEXT("outputs"), Outputs))
	{
		Out.Error = TEXT("malformed reply from unreal_torch");
		return Out;
	}
	TSet<FString> PerRow;
	const TArray<TSharedPtr<FJsonValue>>* PerRowValues = nullptr;
	if (Reply->TryGetArrayField(TEXT("per_row"), PerRowValues))
	{
		for (const TSharedPtr<FJsonValue>& V : *PerRowValues)
		{
			PerRow.Add(V->AsString());
		}
	}
	double Batch = 0.0, Ms = 0.0;
	Reply->TryGetNumberField(TEXT("batch"), Batch);
	Reply->TryGetNumberField(TEXT("ms"), Ms);
	Out.bSuccess = true;
	Out.BatchSize = (int32)Batch;
	Out.ModelMs = (float)Ms;
	TSharedRef<FJsonObject> Json = MakeShared<FJsonObject>();
	for (const TPair<FString, TSharedPtr<FJsonValue>>& Field : (*Outputs)->Values)
	{
		const bool bRow = PerRow.Contains(Field.Key);
		const TSharedPtr<FJsonObject>* Desc = nullptr;
		double Index = -1.0;
		if (Field.Value->TryGetObject(Desc) && (*Desc)->TryGetNumberField(TEXT("_tensor"), Index) && Tensors.IsValidIndex((int32)Index))
		{
			const FPyTensor& T = Tensors[(int32)Index];
			Out.Outputs.Add(Field.Key, bRow ? SliceRows(T, Offset, Rows, bBatched) : T);
			continue;
		}
		const TArray<TSharedPtr<FJsonValue>>* Values = nullptr;
		if (bRow && Field.Value->TryGetArray(Values))
		{
			if (!bBatched && Rows == 1 && Values->IsValidIndex((int32)Offset))
			{
				Json->SetField(Field.Key, (*Values)[(int32)Offset]);
			}
			else
			{
				TArray<TSharedPtr<FJsonValue>> Part;
				for (int64 i = Offset; i < Offset + Rows && i < Values->Num(); ++i)
				{
					Part.Add((*Values)[(int32)i]);
				}
				Json->SetArrayField(Field.Key, Part);
			}
			continue;
		}
		Json->SetField(Field.Key, Field.Value);
	}
	Out.Json = Json;
	Out.OutputsJson = PythonWire::ToJsonString(MakeShared<FJsonValueObject>(Json));
	return Out;
}

int32 UTorchSubsystem::RegisterBinding(UTorchOutputComponent* Component)
{
	const int32 Id = ++NextBindingId;
	Bindings.Add(Id, Component);
	return Id;
}

void UTorchSubsystem::UnregisterBinding(int32 Id)
{
	Bindings.Remove(Id);
}

void UTorchSubsystem::HandleStreamResult(const FPythonEventData& Event)
{
	const TSharedPtr<FJsonObject>* Obj = nullptr;
	double Binding = 0.0;
	if (Event.Args.Num() == 0 || !Event.Args[0]->TryGetObject(Obj) || !(*Obj)->TryGetNumberField(TEXT("binding"), Binding))
	{
		return;
	}
	UTorchOutputComponent* Component = Bindings.FindRef((int32)Binding).Get();
	if (!Component)
	{
		return;
	}
	FString Error;
	if ((*Obj)->TryGetStringField(TEXT("error"), Error))
	{
		FTorchResult Failed;
		Failed.Error = Error;
		(*Obj)->TryGetStringField(TEXT("traceback"), Failed.Traceback);
		Failed.Backend = TEXT("python");
		Component->HandleResult(Failed);
		return;
	}
	FTorchResult Result = MakeResult(*Obj, Event.Tensors, 0, 1, false);
	double Age = 0.0;
	if ((*Obj)->TryGetNumberField(TEXT("frame_age_ms"), Age))
	{
		Result.FrameAgeMs = (float)Age;
	}
	Component->HandleResult(Result);
}

FString UTorchSubsystem::RunsDir()
{
	return FPaths::ConvertRelativePathToFull(FPaths::ProjectSavedDir() / TEXT("Torch/Runs"));
}

UPythonJob* UTorchSubsystem::StartJob(UTorchModel* Model, const FString& JobName, const FString& ParamsJson, bool bResume, bool bOwnProcess)
{
	UPythonSubsystem* Python = UPythonSubsystem::Get();
	if (!Model || !Python || Model->Script.IsEmpty())
	{
		UPythonJob* Failed = NewObject<UPythonJob>(this);
		Failed->Init(nullptr, JobName, FString());
		Failed->Fail(Model ? TEXT("Python is not available (or the model has no Script)") : TEXT("Start Job: no model"));
		return Failed;
	}
	const FTorchJobSpec* Spec = Model->Jobs.FindByPredicate([&JobName](const FTorchJobSpec& J) { return J.Name == JobName; });
	const bool bDedicated = bOwnProcess || (Spec && Spec->Process == TEXT("dedicated"));
	UPythonService* Service = nullptr;
	if (bDedicated)
	{
		Service = Python->GetService(FString::Printf(TEXT("torch-job:%s#%s"), *Model->Script, *JobName), Model->Script, EPythonServiceProcess::Dedicated, {});
		if (Service && !Service->OnNativeEvent(TEXT("_torch_publish")).IsBound())
		{
			TWeakObjectPtr<UTorchSubsystem> WeakSelf(this);
			TWeakObjectPtr<UTorchModel> WeakModel(Model);
			Service->OnNativeEvent(TEXT("_torch_publish")).AddLambda([WeakSelf, WeakModel](const FPythonEventData& Event)
			{
				if (UTorchSubsystem* Self = WeakSelf.Get())
				{
					Self->ForwardPublish(WeakModel.Get(), Event);
				}
			});
		}
	}
	else
	{
		Service = GetService(Model);
	}
	if (!Service)
	{
		UPythonJob* Failed = NewObject<UPythonJob>(this);
		Failed->Init(nullptr, JobName, FString());
		Failed->Fail(TEXT("Python is not available"));
		return Failed;
	}
	return Service->StartJob(JobName, ParamsJson, bResume, RunsDir());
}

void UTorchSubsystem::ForwardPublish(UTorchModel* Model, const FPythonEventData& Event)
{
	const TSharedPtr<FJsonObject>* Obj = nullptr;
	FString ModelName, File;
	if (!Model || Event.Args.Num() == 0 || !Event.Args[0]->TryGetObject(Obj) || !(*Obj)->TryGetStringField(TEXT("model"), ModelName) || !(*Obj)->TryGetStringField(TEXT("file"), File))
	{
		return;
	}
	if (ModelName != Model->GetModelName() || GetModelState(Model) == ETorchModelState::NotLoaded)
	{
		if (FPaths::GetCleanFilename(File).StartsWith(TEXT("published-")))
		{
			IFileManager::Get().Delete(*File, false, false, true); //Nobody loads it: the model picks up the job's saved weights when it loads
		}
		return;
	}
	if (UPythonService* Service = GetService(Model))
	{
		Service->CallFunction(TEXT("_torch_load_weights"), PythonWire::ToJsonString(TArray<TSharedPtr<FJsonValue>>{
			MakeShared<FJsonValueString>(ModelName), MakeShared<FJsonValueString>(File) }), {}, nullptr);
	}
}

void UTorchSubsystem::LoadStartupModels()
{
	IAssetRegistry& Registry = FModuleManager::LoadModuleChecked<FAssetRegistryModule>(TEXT("AssetRegistry")).Get();
	auto LoadThem = [this, &Registry]()
	{
		TArray<FAssetData> Assets;
		Registry.GetAssetsByClass(UTorchModel::StaticClass()->GetClassPathName(), Assets);
		for (const FAssetData& Asset : Assets)
		{
			FString Load;
			if (Asset.GetTagValue(GET_MEMBER_NAME_CHECKED(UTorchModel, Load), Load) && Load.Contains(TEXT("OnStartup")))
			{
				if (UTorchModel* Model = Cast<UTorchModel>(Asset.GetAsset()))
				{
					Model->AddToRoot(); //Kept loaded, like its Python side
					UE_LOG(LogTorch, Log, TEXT("Loading %s at startup"), *Model->GetPathName());
					LoadModel(Model);
				}
			}
		}
	};
	if (Registry.IsLoadingAssets())
	{
		Registry.OnFilesLoaded().AddWeakLambda(this, LoadThem);
	}
	else
	{
		LoadThem();
	}
}

// ------------------------------------------------------------------------------------------- NNE export

#if WITH_EDITOR
void UTorchSubsystem::ExportToNNE(UTorchModel* Model, TFunction<void(bool, const FString&)> OnDone)
{
	if (!Model || !GetService(Model))
	{
		OnDone(false, Model ? TEXT("Python is not available") : TEXT("no model"));
		return;
	}
	TWeakObjectPtr<UTorchModel> Weak(Model);
	TWeakObjectPtr<UTorchSubsystem> WeakSelf(this);
	auto Start = [WeakSelf, Weak, OnDone]()
	{
		UTorchSubsystem* Self = WeakSelf.Get();
		UTorchModel* M = Weak.Get();
		if (!Self || !M)
		{
			return;
		}
		TArray<FString> Methods = M->ExportMethods;
		if (Methods.Num() == 0)
		{
			for (const FTorchMethodSpec& Spec : M->Methods)
			{
				if (Spec.bBatched)
				{
					Methods.Add(Spec.Name);
				}
			}
		}
		if (Methods.Num() == 0)
		{
			OnDone(false, TEXT("no batch=True methods to export"));
			return;
		}
		Self->EnsureLoadSent(M);
		Self->ExportMethod(Weak, Methods, 0, MakeShared<TArray<FString>>(), MakeShared<bool>(true), OnDone);
	};
	if (Model->Methods.Num() == 0 && Model->ExportMethods.Num() == 0)
	{
		FetchSignature(Model, [Start, OnDone](bool bOk, const FString& Error)
		{
			if (bOk)
			{
				Start();
			}
			else
			{
				OnDone(false, TEXT("the signature: ") + Error);
			}
		});
		return;
	}
	Start();
}

void UTorchSubsystem::ExportMethod(TWeakObjectPtr<UTorchModel> Model, TArray<FString> Methods, int32 Index, TSharedRef<TArray<FString>> Summary, TSharedRef<bool> bAllOk,
	TFunction<void(bool, const FString&)> OnDone)
{
	UTorchModel* M = Model.Get();
	if (!M)
	{
		return;
	}
	if (!Methods.IsValidIndex(Index))
	{
		//Persist the asset (a real one): its NNE Methods point at the new model data
		UPackage* Package = M->GetOutermost();
		if (Package != GetTransientPackage() && FPackageName::IsValidLongPackageName(Package->GetName()))
		{
			M->MarkPackageDirty();
			FSavePackageArgs Args;
			Args.TopLevelFlags = RF_Public | RF_Standalone;
			UPackage::SavePackage(Package, M, *FPackageName::LongPackageNameToFilename(Package->GetName(), FPackageName::GetAssetPackageExtension()), Args);
		}
		OnDone(*bAllOk, FString::Join(*Summary, TEXT("; ")));
		return;
	}
	UPythonService* Service = GetService(M);
	const FString Method = Methods[Index];
	TSharedRef<FJsonObject> Options = MakeShared<FJsonObject>();
	Options->SetNumberField(TEXT("static_batch"), M->ExportStaticBatch);
	Options->SetStringField(TEXT("out_dir"), FPaths::ConvertRelativePathToFull(FPaths::ProjectSavedDir() / TEXT("Torch/Export") / M->GetName()));
	const FString Args = PythonWire::ToJsonString(TArray<TSharedPtr<FJsonValue>>{
		MakeShared<FJsonValueString>(M->GetModelName()), MakeShared<FJsonValueString>(Method), MakeShared<FJsonValueObject>(Options) });
	TWeakObjectPtr<UTorchSubsystem> WeakSelf(this);
	Service->CallFunction(TEXT("_torch_export"), Args, {}, [WeakSelf, Model, Methods, Index, Summary, bAllOk, OnDone, Method](const FPythonCallResult& R)
	{
		UTorchSubsystem* Self = WeakSelf.Get();
		UTorchModel* M = Model.Get();
		if (!Self || !M)
		{
			return;
		}
		const TSharedPtr<FJsonObject>* Desc = nullptr;
		if (!R.bSuccess || !R.Result.IsValid() || !R.Result->TryGetObject(Desc))
		{
			*bAllOk = false;
			Summary->Add(FString::Printf(TEXT("%s: %s"), *Method, R.bSuccess ? TEXT("malformed reply") : *R.Error));
		}
		else
		{
			FString Line;
			if (!Self->ImportExport(M, *Desc, Line))
			{
				*bAllOk = false;
			}
			Summary->Add(Line);
		}
		Self->ExportMethod(Model, Methods, Index + 1, Summary, bAllOk, OnDone);
	});
}

namespace TorchSubsystemPrivate
{
	/** An ONNX file as NNE model data: an asset next to the Torch Model (<Asset>_<Suffix>), or a transient object for
	 transient models. */
	static UNNEModelData* MakeModelData(UTorchModel* Model, const FString& Suffix, const FString& File, FString& OutError)
	{
		TArray64<uint8> Bytes;
		if (!FFileHelper::LoadFileToArray(Bytes, *File))
		{
			OutError = TEXT("can't read ") + File;
			return nullptr;
		}
		const FString AssetName = Model->GetName() + TEXT("_") + Suffix;
		UPackage* ModelPackage = Model->GetOutermost();
		if (ModelPackage == GetTransientPackage() || !FPackageName::IsValidLongPackageName(ModelPackage->GetName()))
		{
			UNNEModelData* Data = NewObject<UNNEModelData>(Model, MakeUniqueObjectName(Model, UNNEModelData::StaticClass(), *AssetName));
			Data->Init(TEXT("onnx"), Bytes);
			return Data;
		}
		const FString PackageName = FPackageName::GetLongPackagePath(ModelPackage->GetName()) / AssetName;
		UPackage* Package = CreatePackage(*PackageName);
		Package->FullyLoad();
		UNNEModelData* Data = FindObject<UNNEModelData>(Package, *AssetName);
		const bool bNew = Data == nullptr;
		if (bNew)
		{
			Data = NewObject<UNNEModelData>(Package, *AssetName, RF_Public | RF_Standalone | RF_Transactional);
		}
		Data->Modify();
		Data->Init(TEXT("onnx"), Bytes);
		if (bNew)
		{
			FAssetRegistryModule::AssetCreated(Data);
		}
		Package->MarkPackageDirty();
		FSavePackageArgs Args;
		Args.TopLevelFlags = RF_Public | RF_Standalone;
		if (!UPackage::SavePackage(Package, Data, *FPackageName::LongPackageNameToFilename(PackageName, FPackageName::GetAssetPackageExtension()), Args))
		{
			OutError = TEXT("couldn't save ") + PackageName;
		}
		return Data;
	}
}

bool UTorchSubsystem::ImportExport(UTorchModel* Model, const TSharedPtr<FJsonObject>& Desc, FString& OutSummary)
{
	using namespace TorchSubsystemPrivate;
	const FString Method = Desc->GetStringField(TEXT("method"));
	const TSharedPtr<FJsonObject>* Files = nullptr;
	FString DynamicFile, StaticFile, ParityFile;
	if (!Desc->TryGetObjectField(TEXT("files"), Files) || !(*Files)->TryGetStringField(TEXT("dynamic"), DynamicFile))
	{
		OutSummary = Method + TEXT(": the export wrote no files");
		return false;
	}
	(*Files)->TryGetStringField(TEXT("static"), StaticFile);
	(*Files)->TryGetStringField(TEXT("parity"), ParityFile);
	int32 StaticBatch = 0;
	Desc->TryGetNumberField(TEXT("static_batch"), StaticBatch);

	FString Error;
	FTorchNNEMethod Entry;
	Entry.Method = Method;
	Entry.SpecJson = PythonWire::ToJsonString(MakeShared<FJsonValueObject>(Desc));
	Entry.Model = MakeModelData(Model, Method, DynamicFile, Error);
	if (!StaticFile.IsEmpty() && StaticBatch > 0)
	{
		Entry.StaticModel = MakeModelData(Model, FString::Printf(TEXT("%s_b%d"), *Method, StaticBatch), StaticFile, Error);
		Entry.StaticBatch = Entry.StaticModel ? StaticBatch : 0;
	}
	if (!Entry.Model)
	{
		OutSummary = FString::Printf(TEXT("%s: %s"), *Method, *Error);
		return false;
	}
	Model->Modify();
	Model->NNEMethods.RemoveAll([&Method](const FTorchNNEMethod& M) { return M.Method == Method; });
	Model->NNEMethods.Add(Entry);
	FTorchNNE::Get().Reset(Model);

	//Parity: the export's reference rows through each runtime NNE made for it, against torch's outputs
	TMap<FString, FPyTensor> Reference;
	TArray<FString> Report;
	bool bParityOk = true;
	if (!ParityFile.IsEmpty() && FTorchNNE::ReadSafetensors(ParityFile, Reference, Error))
	{
		TArray<FPyTensor> Inputs;
		const TArray<TSharedPtr<FJsonValue>>* InputSpecs = nullptr;
		if (Desc->TryGetArrayField(TEXT("inputs"), InputSpecs))
		{
			for (const TSharedPtr<FJsonValue>& V : *InputSpecs)
			{
				if (const FPyTensor* T = Reference.Find(TEXT("in.") + V->AsObject()->GetStringField(TEXT("name"))))
				{
					Inputs.Add(*T);
				}
			}
		}
		const UTorchSettings* Settings = GetDefault<UTorchSettings>();
		TArray<FString> Runtimes = Settings->NNECPURuntimes;
		Runtimes.Append(Settings->NNEGPURuntimes);
		const ETorchNNERuntime Preference = Model->NNERuntime;
		Model->NNERuntime = ETorchNNERuntime::Auto; //Check every runtime there is
		for (const FString& Runtime : Runtimes)
		{
			const FTorchNNE::FResult R = FTorchNNE::Get().RunNow(Model, Method, Inputs, Runtime);
			if (!R.bSuccess)
			{
				if (!R.Error.Contains(TEXT("isn't one of its runtimes")))
				{
					Report.Add(FString::Printf(TEXT("%s failed: %s"), *Runtime, *R.Error));
					bParityOk = false;
				}
				continue;
			}
			float MaxDiff = 0.f, MaxAbs = 1.f;
			for (int32 k = 0; k < R.Names.Num(); ++k)
			{
				const FPyTensor* Expected = Reference.Find(TEXT("out.") + R.Names[k]);
				if (!Expected)
				{
					continue;
				}
				const TArray<float> A = R.Outputs[k].ToFloats();
				const TArray<float> B = Expected->ToFloats();
				for (int32 i = 0; i < FMath::Min(A.Num(), B.Num()); ++i)
				{
					MaxDiff = FMath::Max(MaxDiff, FMath::Abs(A[i] - B[i]));
					MaxAbs = FMath::Max(MaxAbs, FMath::Abs(B[i]));
				}
			}
			if (Entry.ParityMaxDiff < 0.f)
			{
				Model->NNEMethods.Last().ParityMaxDiff = MaxDiff;
				Entry.ParityMaxDiff = MaxDiff;
			}
			Report.Add(FString::Printf(TEXT("%s max diff %.2g"), *Runtime, MaxDiff));
			bParityOk = bParityOk && MaxDiff <= 1e-2f * MaxAbs;
		}
		Model->NNEMethods.Last().ParityMaxDiff = Entry.ParityMaxDiff;
		FTorchNNE::Get().Reset(Model);
		Model->NNERuntime = Preference;
	}
	else
	{
		Report.Add(TEXT("not checked: ") + Error);
	}
	OutSummary = FString::Printf(TEXT("%s -> %s%s (%s)"), *Method, *GetNameSafe(Entry.Model), Entry.StaticModel ? *FString::Printf(TEXT(" + %s"), *GetNameSafe(Entry.StaticModel)) : TEXT(""),
		*FString::Join(Report, TEXT(", ")));
	UE_LOG(LogTorch, Log, TEXT("%s: NNE export %s"), *Model->GetName(), *OutSummary);
	return bParityOk;
}
#endif
