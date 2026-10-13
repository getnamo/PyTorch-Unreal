#include "TorchSubsystem.h"
#include "PyTorch.h"
#include "PythonJob.h"
#include "PythonService.h"
#include "PythonSubsystem.h"
#include "TorchModel.h"
#include "TorchOutputComponent.h"
#include "AssetRegistry/AssetRegistryModule.h"
#include "AssetRegistry/IAssetRegistry.h"
#include "Dom/JsonObject.h"
#include "Engine/Engine.h"
#include "Misc/CoreDelegates.h"
#include "Misc/Paths.h"
#include "Modules/ModuleManager.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

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
	if (!Model || Model->Script.IsEmpty())
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
	EnsureLoadSent(Model); //Calls to the script run in order: the load is first
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

void UTorchSubsystem::SendGroup(UTorchModel* Model, const FString& Method, TArray<FRequest>&& Requests)
{
	using namespace TorchSubsystemPrivate;
	UPythonService* Service = GetService(Model);
	if (!Service)
	{
		Fail(Requests, TEXT("Python is not available"));
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
	First.Inputs.GetKeys(Names);
	Names.Sort();
	TArray<FPyTensor> Tensors;
	for (const FString& Name : Names)
	{
		const FPyTensor& T0 = First.Inputs[Name];
		FPyTensor Stacked;
		Stacked.DType = T0.DType;
		int64 Total = 0;
		for (const FRequest& R : Requests)
		{
			Stacked.Data.Append(R.Inputs[Name].Data);
			Total += R.Rows;
		}
		Stacked.Shape = { Total };
		Stacked.Shape.Append(RowShape(T0, First.Options.bBatched));
		TSharedRef<FJsonObject> Desc = MakeShared<FJsonObject>();
		Desc->SetNumberField(TEXT("_tensor"), Tensors.Num());
		Desc->SetStringField(TEXT("dtype"), FPyTensor::DTypeName(Stacked.DType));
		TArray<TSharedPtr<FJsonValue>> ShapeValues;
		for (int64 D : Stacked.Shape)
		{
			ShapeValues.Add(MakeShared<FJsonValueNumber>((double)D));
		}
		Desc->SetArrayField(TEXT("shape"), ShapeValues);
		InputsObj->SetObjectField(Name, Desc);
		Tensors.Add(MoveTemp(Stacked));
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
		return; //Not loaded in the game: it picks the weights up when it loads (Saved/Torch/Models)
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
