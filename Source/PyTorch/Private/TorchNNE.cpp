#include "TorchNNE.h"
#include "PyTorch.h"
#include "TorchModel.h"
#include "TorchSettings.h"
#include "TorchSubsystem.h"
#include "NNE.h"
#include "NNEModelData.h"
#include "NNERuntimeCPU.h"
#include "NNERuntimeGPU.h"
#include "NNERuntimeRunSync.h"
#include "NNETypes.h"
#include "Async/Async.h"
#include "Dom/JsonObject.h"
#include "Misc/FileHelper.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

namespace TorchNNEPrivate
{
	static bool ToPyDType(ENNETensorDataType T, EPyDType& Out)
	{
		switch (T)
		{
		case ENNETensorDataType::Float: Out = EPyDType::Float32; return true;
		case ENNETensorDataType::Double: Out = EPyDType::Float64; return true;
		case ENNETensorDataType::Half: Out = EPyDType::Float16; return true;
		case ENNETensorDataType::Int8: Out = EPyDType::Int8; return true;
		case ENNETensorDataType::UInt8: Out = EPyDType::UInt8; return true;
		case ENNETensorDataType::Int16: Out = EPyDType::Int16; return true;
		case ENNETensorDataType::UInt16: Out = EPyDType::UInt16; return true;
		case ENNETensorDataType::Int32: Out = EPyDType::Int32; return true;
		case ENNETensorDataType::UInt32: Out = EPyDType::UInt32; return true;
		case ENNETensorDataType::Int64: Out = EPyDType::Int64; return true;
		case ENNETensorDataType::UInt64: Out = EPyDType::UInt64; return true;
		case ENNETensorDataType::Boolean: Out = EPyDType::Bool; return true;
		default: return false;
		}
	}

	static int64 RowBytes(const FPyTensor& T)
	{
		int64 N = 1;
		for (int32 i = 1; i < T.Shape.Num(); ++i)
		{
			N *= T.Shape[i];
		}
		return N * FPyTensor::ItemSize(T.DType);
	}

	/** Rows [Start, Start + Count) of T; rows past its end repeat its last row (padding a static batch). */
	static FPyTensor Rows(const FPyTensor& T, int64 Start, int64 Count)
	{
		FPyTensor Out;
		Out.DType = T.DType;
		Out.Shape = T.Shape;
		Out.Shape[0] = Count;
		const int64 Row = RowBytes(T);
		const int64 Have = T.Shape.Num() ? T.Shape[0] : 0;
		Out.Data.SetNumUninitialized(Row * Count);
		for (int64 r = 0; r < Count; ++r)
		{
			const int64 From = FMath::Min(Start + r, Have - 1);
			FMemory::Memcpy(Out.Data.GetData() + r * Row, T.Data.GetData() + From * Row, Row);
		}
		return Out;
	}

	static void AppendRows(FPyTensor& Into, const FPyTensor& From, int64 Count)
	{
		if (Into.Shape.Num() == 0)
		{
			Into.DType = From.DType;
			Into.Shape = From.Shape;
			Into.Shape[0] = 0;
		}
		Into.Data.Append(From.Data.GetData(), RowBytes(From) * Count);
		Into.Shape[0] += Count;
	}

	/** PyTorch's antialiased bilinear weights (interpolate(..., antialias=True)): a triangle filter widened by the
	 scale when shrinking; plain bilinear when growing. */
	struct FTaps
	{
		int32 Start = 0;
		TArray<float> Weights;
	};

	static TArray<FTaps> ResizeWeights(int32 In, int32 Out)
	{
		TArray<FTaps> Taps;
		Taps.SetNum(Out);
		const double Scale = (double)In / Out;
		const double Support = Scale >= 1.0 ? Scale : 1.0;
		const double InvScale = Scale >= 1.0 ? 1.0 / Scale : 1.0;
		for (int32 i = 0; i < Out; ++i)
		{
			const double Center = Scale * (i + 0.5);
			const int32 Min = FMath::Max((int32)(Center - Support + 0.5), 0);
			const int32 Max = FMath::Min((int32)(Center + Support + 0.5), In);
			FTaps& T = Taps[i];
			T.Start = Min;
			double Sum = 0.0;
			for (int32 j = Min; j < Max; ++j)
			{
				const double W = FMath::Max(0.0, 1.0 - FMath::Abs((j - Center + 0.5) * InvScale));
				T.Weights.Add((float)W);
				Sum += W;
			}
			for (float& W : T.Weights)
			{
				W = Sum > 0.0 ? (float)(W / Sum) : 0.f;
			}
		}
		return Taps;
	}

	/** One plane (H x W) resized to (OutH x OutW), separably. */
	static void Resize(const float* Src, int32 H, int32 W, float* Dst, int32 OutH, int32 OutW, const TArray<FTaps>& Rows, const TArray<FTaps>& Cols)
	{
		TArray<float> Tmp;
		Tmp.SetNumUninitialized(H * OutW);
		for (int32 y = 0; y < H; ++y)
		{
			for (int32 x = 0; x < OutW; ++x)
			{
				const FTaps& T = Cols[x];
				float V = 0.f;
				for (int32 k = 0; k < T.Weights.Num(); ++k)
				{
					V += T.Weights[k] * Src[y * W + T.Start + k];
				}
				Tmp[y * OutW + x] = V;
			}
		}
		for (int32 y = 0; y < OutH; ++y)
		{
			const FTaps& T = Rows[y];
			for (int32 x = 0; x < OutW; ++x)
			{
				float V = 0.f;
				for (int32 k = 0; k < T.Weights.Num(); ++k)
				{
					V += T.Weights[k] * Tmp[(T.Start + k) * OutW + x];
				}
				Dst[y * OutW + x] = V;
			}
		}
	}

	static TArray<int64> ShapeOf(const FJsonObject& Spec, const TCHAR* Field)
	{
		TArray<int64> Shape;
		const TArray<TSharedPtr<FJsonValue>>* Values = nullptr;
		if (Spec.TryGetArrayField(Field, Values))
		{
			for (const TSharedPtr<FJsonValue>& V : *Values)
			{
				Shape.Add((int64)V->AsNumber());
			}
		}
		return Shape;
	}
}

// ------------------------------------------------------------------------------------------------- state

struct FTorchNNE::FInstance
{
	TSharedPtr<UE::NNE::IModelInstanceRunSync> Instance;
	FString Runtime;
	bool bGpu = false;
	/** > 0: a static-batch model of this many rows. */
	int32 StaticBatch = 0;
	TArray<TArray<uint32>> Shapes;
};

struct FTorchNNE::FState
{
	/** The instances aren't thread-safe: one run at a time. */
	FCriticalSection Lock;
	FString Label;
	TArray<TSharedPtr<FJsonObject>> InputSpecs;
	/** Per-row output shapes from the export: NNE knows dynamic models' output shapes only after a run. */
	TMap<FString, TArray<int64>> OutputRowShapes;
	TArray<FInstance> Instances;
	/** Runtimes in the order to consider; Chosen once picked (Auto: the fastest on the first run). */
	TArray<FString> RuntimeOrder;
	FString Chosen;
};

FTorchNNE& FTorchNNE::Get()
{
	static FTorchNNE Instance;
	return Instance;
}

namespace TorchNNEPrivate
{
	static bool CreateInstance(UNNEModelData* Data, const FString& RuntimeName, bool bGpu, TSharedPtr<UE::NNE::IModelInstanceRunSync>& Out, FString& Error)
	{
		if (bGpu)
		{
			TWeakInterfacePtr<INNERuntimeGPU> Runtime = UE::NNE::GetRuntime<INNERuntimeGPU>(RuntimeName);
			if (!Runtime.IsValid())
			{
				Error = RuntimeName + TEXT(" isn't available");
				return false;
			}
			if (Runtime->CanCreateModelGPU(Data) != INNERuntimeGPU::ECanCreateModelGPUStatus::Ok)
			{
				Error = RuntimeName + TEXT(" can't create this model");
				return false;
			}
			TSharedPtr<UE::NNE::IModelGPU> Model = Runtime->CreateModelGPU(Data);
			Out = Model.IsValid() ? Model->CreateModelInstanceGPU() : nullptr;
		}
		else
		{
			TWeakInterfacePtr<INNERuntimeCPU> Runtime = UE::NNE::GetRuntime<INNERuntimeCPU>(RuntimeName);
			if (!Runtime.IsValid())
			{
				Error = RuntimeName + TEXT(" isn't available");
				return false;
			}
			if (Runtime->CanCreateModelCPU(Data) != INNERuntimeCPU::ECanCreateModelCPUStatus::Ok)
			{
				Error = RuntimeName + TEXT(" can't create this model");
				return false;
			}
			TSharedPtr<UE::NNE::IModelCPU> Model = Runtime->CreateModelCPU(Data);
			Out = Model.IsValid() ? Model->CreateModelInstanceCPU() : nullptr;
		}
		if (!Out.IsValid())
		{
			Error = RuntimeName + TEXT(": creating the model instance failed");
			return false;
		}
		return true;
	}
}

TSharedPtr<FTorchNNE::FState> FTorchNNE::FindOrCreate(UTorchModel* Model, const FString& Method, FString& OutError)
{
	using namespace TorchNNEPrivate;
	check(IsInGameThread());
	const FTorchNNEMethod* M = Model ? Model->FindNNEMethod(Method) : nullptr;
	if (!M)
	{
		OutError = FString::Printf(TEXT("%s has no NNE export of '%s' (Export To NNE on the Torch Model)"), Model ? *Model->GetName() : TEXT("(no model)"), *Method);
		return nullptr;
	}
	const FString Key = FString::Printf(TEXT("%s|%s|%d|%p"), *UTorchSubsystem::ModelKey(Model), *Method, (int32)Model->NNERuntime, M->Model.Get());
	if (const TSharedPtr<FState>* Found = States.Find(Key))
	{
		return *Found;
	}
	TSharedPtr<FJsonObject> Spec;
	if (!FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(M->SpecJson), Spec) || !Spec.IsValid())
	{
		OutError = FString::Printf(TEXT("%s.%s: the NNE export's description is missing (export again)"), *Model->GetName(), *Method);
		return nullptr;
	}
	TSharedPtr<FState> State = MakeShared<FState>();
	State->Label = Model->GetModelName() + TEXT(".") + Method;
	const TArray<TSharedPtr<FJsonValue>>* Inputs = nullptr;
	if (Spec->TryGetArrayField(TEXT("inputs"), Inputs))
	{
		for (const TSharedPtr<FJsonValue>& V : *Inputs)
		{
			State->InputSpecs.Add(V->AsObject());
		}
	}
	const TArray<TSharedPtr<FJsonValue>>* Outputs = nullptr;
	if (Spec->TryGetArrayField(TEXT("outputs"), Outputs))
	{
		for (const TSharedPtr<FJsonValue>& V : *Outputs)
		{
			State->OutputRowShapes.Add(V->AsObject()->GetStringField(TEXT("name")), ShapeOf(*V->AsObject(), TEXT("shape")));
		}
	}

	const UTorchSettings* Settings = GetDefault<UTorchSettings>();
	const bool bWantCpu = Model->NNERuntime != ETorchNNERuntime::GPU;
	const bool bWantGpu = Model->NNERuntime != ETorchNNERuntime::CPU;
	TArray<FString> Errors;
	auto AddRuntime = [&](const TArray<FString>& Names, bool bGpu) -> bool
	{
		for (const FString& Name : Names)
		{
			FInstance Dynamic;
			FString Error;
			if (!CreateInstance(M->Model, Name, bGpu, Dynamic.Instance, Error))
			{
				Errors.Add(Error);
				continue;
			}
			Dynamic.Runtime = Name;
			Dynamic.bGpu = bGpu;
			State->Instances.Add(MoveTemp(Dynamic));
			if (bGpu && M->StaticModel && M->StaticBatch > 0)
			{
				FInstance Static;
				if (CreateInstance(M->StaticModel, Name, true, Static.Instance, Error))
				{
					Static.Runtime = Name;
					Static.bGpu = true;
					Static.StaticBatch = M->StaticBatch;
					State->Instances.Add(MoveTemp(Static));
				}
				else
				{
					Errors.Add(Error);
				}
			}
			State->RuntimeOrder.Add(Name);
			return true;
		}
		return false;
	};
	if (bWantGpu)
	{
		AddRuntime(Settings->NNEGPURuntimes, true);
	}
	if (bWantCpu || State->Instances.Num() == 0)
	{
		AddRuntime(Settings->NNECPURuntimes, false);
	}
	if (!bWantCpu && State->Instances.Num() == 0)
	{
		AddRuntime(Settings->NNEGPURuntimes, true);
	}
	if (State->Instances.Num() == 0)
	{
		OutError = FString::Printf(TEXT("%s: no NNE runtime could create it (%s)"), *State->Label, *FString::Join(Errors, TEXT("; ")));
		return nullptr;
	}
	if (State->RuntimeOrder.Num() == 1)
	{
		State->Chosen = State->RuntimeOrder[0];
	}
	States.Add(Key, State);
	return State;
}

bool FTorchNNE::Prepare(UTorchModel* Model, const FString& Method, FString& OutError)
{
	return FindOrCreate(Model, Method, OutError).IsValid();
}

void FTorchNNE::Reset(const UTorchModel* Model)
{
	const FString Prefix = UTorchSubsystem::ModelKey(Model) + TEXT("|");
	for (auto It = States.CreateIterator(); It; ++It)
	{
		if (It.Key().StartsWith(Prefix))
		{
			It.RemoveCurrent();
		}
	}
}

FString FTorchNNE::GetRuntime(const UTorchModel* Model, const FString& Method) const
{
	const FString Prefix = UTorchSubsystem::ModelKey(Model) + TEXT("|") + Method + TEXT("|");
	for (const TPair<FString, TSharedPtr<FState>>& Pair : States)
	{
		if (Pair.Key.StartsWith(Prefix))
		{
			FScopeLock Lock(&Pair.Value->Lock);
			return Pair.Value->Chosen;
		}
	}
	return FString();
}

// --------------------------------------------------------------------------------------------- running

namespace TorchNNEPrivate
{
	static bool RunInstance(FTorchNNE::FResult& Out, TSharedPtr<UE::NNE::IModelInstanceRunSync> Instance, TArray<TArray<uint32>>& LastShapes, const TArray<FPyTensor>& Inputs,
		const TMap<FString, TArray<int64>>& OutputRowShapes)
	{
		TArray<TArray<uint32>> Shapes;
		for (const FPyTensor& T : Inputs)
		{
			TArray<uint32>& S = Shapes.AddDefaulted_GetRef();
			for (int64 D : T.Shape)
			{
				S.Add((uint32)D);
			}
		}
		if (Shapes != LastShapes)
		{
			TArray<UE::NNE::FTensorShape> NNEShapes;
			for (const TArray<uint32>& S : Shapes)
			{
				NNEShapes.Add(UE::NNE::FTensorShape::Make(S));
			}
			if (Instance->SetInputTensorShapes(NNEShapes) != UE::NNE::EResultStatus::Ok)
			{
				Out.Error = TEXT("the inputs' shapes don't fit the model");
				LastShapes.Reset();
				return false;
			}
			LastShapes = Shapes;
		}
		const TConstArrayView<UE::NNE::FTensorDesc> Descs = Instance->GetOutputTensorDescs();
		const TConstArrayView<UE::NNE::FTensorShape> OutShapes = Instance->GetOutputTensorShapes();
		const int64 Rows = Inputs.Num() && Inputs[0].Shape.Num() ? Inputs[0].Shape[0] : 1;
		Out.Names.Reset();
		Out.Outputs.Reset();
		TArray<UE::NNE::FTensorBindingCPU> OutBindings;
		for (int32 i = 0; i < Descs.Num(); ++i)
		{
			FPyTensor& T = Out.Outputs.AddDefaulted_GetRef();
			if (!ToPyDType(Descs[i].GetDataType(), T.DType))
			{
				Out.Error = FString::Printf(TEXT("output '%s' has a data type Unreal tensors don't have"), *Descs[i].GetName());
				return false;
			}
			if (OutShapes.Num() == Descs.Num())
			{
				for (uint32 D : OutShapes[i].GetData())
				{
					T.Shape.Add(D);
				}
			}
			else if (const TArray<int64>* RowShape = OutputRowShapes.Find(Descs[i].GetName()))
			{
				T.Shape = { Rows };
				T.Shape.Append(*RowShape);
			}
			else
			{
				Out.Error = FString::Printf(TEXT("output '%s': its shape isn't known"), *Descs[i].GetName());
				return false;
			}
			int64 Volume = 1;
			for (int64 D : T.Shape)
			{
				Volume *= D;
			}
			T.Data.SetNumZeroed(Volume * FPyTensor::ItemSize(T.DType));
			Out.Names.Add(Descs[i].GetName());
		}
		for (FPyTensor& T : Out.Outputs)
		{
			OutBindings.Add({ T.Data.GetData(), (uint64)T.Data.Num() });
		}
		TArray<UE::NNE::FTensorBindingCPU> InBindings;
		for (const FPyTensor& T : Inputs)
		{
			InBindings.Add({ const_cast<uint8*>(T.Data.GetData()), (uint64)T.Data.Num() });
		}
		if (Instance->RunSync(InBindings, OutBindings) != UE::NNE::EResultStatus::Ok)
		{
			Out.Error = TEXT("the NNE run failed");
			return false;
		}
		return true;
	}

	/** One runtime's run: a static-batch model for several rows (padded / in chunks), else the dynamic one. */
	static FTorchNNE::FResult RunRuntime(TArray<FTorchNNE::FInstance*>& Group, const TArray<FPyTensor>& Inputs, const TMap<FString, TArray<int64>>& OutputRowShapes)
	{
		FTorchNNE::FResult Out;
		Out.Runtime = Group.Num() ? Group[0]->Runtime : FString();
		FTorchNNE::FInstance* Dynamic = nullptr;
		FTorchNNE::FInstance* Static = nullptr;
		for (FTorchNNE::FInstance* I : Group)
		{
			(I->StaticBatch > 0 ? Static : Dynamic) = I;
		}
		const int64 Total = Inputs.Num() && Inputs[0].Shape.Num() ? Inputs[0].Shape[0] : 1;
		const double Start = FPlatformTime::Seconds();
		if (Static && (Total > 1 || !Dynamic))
		{
			const int64 B = Static->StaticBatch;
			TArray<FPyTensor> Merged;
			TArray<FString> Names;
			for (int64 First = 0; First < Total; First += B)
			{
				const int64 N = FMath::Min(B, Total - First);
				TArray<FPyTensor> Chunk;
				for (const FPyTensor& T : Inputs)
				{
					Chunk.Add(Rows(T, First, B));
				}
				FTorchNNE::FResult Part;
				if (!RunInstance(Part, Static->Instance, Static->Shapes, Chunk, OutputRowShapes))
				{
					Out.Error = Part.Error;
					return Out;
				}
				if (Merged.Num() == 0)
				{
					Merged.SetNum(Part.Outputs.Num());
					Names = Part.Names;
				}
				for (int32 k = 0; k < Part.Outputs.Num(); ++k)
				{
					AppendRows(Merged[k], Part.Outputs[k], N);
				}
			}
			Out.Names = MoveTemp(Names);
			Out.Outputs = MoveTemp(Merged);
		}
		else if (!RunInstance(Out, Dynamic->Instance, Dynamic->Shapes, Inputs, OutputRowShapes))
		{
			return Out;
		}
		Out.Ms = (float)((FPlatformTime::Seconds() - Start) * 1000.0);
		Out.bSuccess = true;
		return Out;
	}
}

FTorchNNE::FResult FTorchNNE::Execute(FState& S, const TArray<FPyTensor>& Inputs, const FString& ForceRuntime)
{
	using namespace TorchNNEPrivate;
	FScopeLock Lock(&S.Lock);
	auto GroupOf = [&S](const FString& Runtime)
	{
		TArray<FInstance*> Group;
		for (FInstance& I : S.Instances)
		{
			if (I.Runtime == Runtime)
			{
				Group.Add(&I);
			}
		}
		return Group;
	};
	if (!ForceRuntime.IsEmpty())
	{
		TArray<FInstance*> Group = GroupOf(ForceRuntime);
		if (Group.Num() == 0)
		{
			FResult Out;
			Out.Error = FString::Printf(TEXT("%s: %s isn't one of its runtimes (%s)"), *S.Label, *ForceRuntime, *FString::Join(S.RuntimeOrder, TEXT(", ")));
			return Out;
		}
		return RunRuntime(Group, Inputs, S.OutputRowShapes);
	}
	if (S.Chosen.IsEmpty())
	{
		//Auto: time each runtime on these inputs (a first run warms it up) and keep the fastest
		FResult Best;
		double BestMs = TNumericLimits<double>::Max();
		TArray<FString> Report;
		for (const FString& Runtime : S.RuntimeOrder)
		{
			TArray<FInstance*> Group = GroupOf(Runtime);
			FResult First = RunRuntime(Group, Inputs, S.OutputRowShapes);
			if (!First.bSuccess)
			{
				Report.Add(FString::Printf(TEXT("%s failed: %s"), *Runtime, *First.Error));
				continue;
			}
			double Ms = TNumericLimits<double>::Max();
			for (int32 i = 0; i < 3; ++i)
			{
				const FResult Again = RunRuntime(Group, Inputs, S.OutputRowShapes);
				Ms = FMath::Min(Ms, (double)Again.Ms);
			}
			Report.Add(FString::Printf(TEXT("%s %.3f ms"), *Runtime, Ms));
			if (Ms < BestMs)
			{
				BestMs = Ms;
				Best = MoveTemp(First);
				S.Chosen = Runtime;
			}
		}
		if (S.Chosen.IsEmpty())
		{
			Best.Error = FString::Printf(TEXT("%s: no NNE runtime ran it (%s)"), *S.Label, *FString::Join(Report, TEXT("; ")));
			return Best;
		}
		UE_LOG(LogTorch, Log, TEXT("%s: NNE runtime %s (timed on %lld rows: %s)"), *S.Label, *S.Chosen, Inputs.Num() && Inputs[0].Shape.Num() ? Inputs[0].Shape[0] : 0, *FString::Join(Report, TEXT(", ")));
		S.Instances.RemoveAll([&S](const FInstance& I) { return I.Runtime != S.Chosen; }); //Free the others (GPU memory)
		return Best;
	}
	TArray<FInstance*> Group = GroupOf(S.Chosen);
	return RunRuntime(Group, Inputs, S.OutputRowShapes);
}

FTorchNNE::FResult FTorchNNE::RunNow(UTorchModel* Model, const FString& Method, const TArray<FPyTensor>& ModelInputs, const FString& ForceRuntime)
{
	FString Error;
	TSharedPtr<FState> State = FindOrCreate(Model, Method, Error);
	if (!State)
	{
		FResult Out;
		Out.Error = Error;
		return Out;
	}
	return Execute(*State, ModelInputs, ForceRuntime);
}

void FTorchNNE::Run(UTorchModel* Model, const FString& Method, TMap<FString, FPyTensor>&& Inputs, int64 Rows, const TMap<FString, FString>& Layouts,
	TFunction<void(FResult&&)> OnDone)
{
	FString Error;
	TSharedPtr<FState> State = FindOrCreate(Model, Method, Error);
	if (!State)
	{
		FResult Out;
		Out.Error = Error;
		OnDone(MoveTemp(Out));
		return;
	}
	Async(EAsyncExecution::ThreadPool, [State, Inputs = MoveTemp(Inputs), Rows, Layouts, OnDone = MoveTemp(OnDone)]() mutable
	{
		FResult Out;
		TArray<FPyTensor> ModelInputs;
		for (const TSharedPtr<FJsonObject>& Spec : State->InputSpecs)
		{
			const FString Name = Spec->GetStringField(TEXT("name"));
			const FPyTensor* In = Inputs.Find(Name);
			if (!In)
			{
				Out.Error = FString::Printf(TEXT("%s: missing input '%s'"), *State->Label, *Name);
				break;
			}
			FPyTensor& Converted = ModelInputs.AddDefaulted_GetRef();
			FString Error;
			if (!ConvertInput(*Spec, *In, Rows, Layouts.FindRef(Name), Converted, Error))
			{
				Out.Error = FString::Printf(TEXT("%s: input '%s': %s"), *State->Label, *Name, *Error);
				break;
			}
		}
		if (Out.Error.IsEmpty())
		{
			Out = Execute(*State, ModelInputs, FString());
		}
		AsyncTask(ENamedThreads::GameThread, [Out = MoveTemp(Out), OnDone = MoveTemp(OnDone)]() mutable { OnDone(MoveTemp(Out)); });
	});
}

// ---------------------------------------------------------------------------------------- input conversion

bool FTorchNNE::ConvertInput(const FJsonObject& Spec, const FPyTensor& In, int64 Rows, const FString& Layout, FPyTensor& Out, FString& OutError)
{
	using namespace TorchNNEPrivate;
	const FString Kind = Spec.GetStringField(TEXT("kind"));
	if (Kind != TEXT("image"))
	{
		const TArray<int64> RowShape = ShapeOf(Spec, TEXT("shape"));
		int64 RowVolume = 1;
		for (int64 D : RowShape)
		{
			RowVolume *= D;
		}
		if (In.NumElements() != Rows * RowVolume)
		{
			OutError = FString::Printf(TEXT("expected %lld rows of %lld values, got %lld values"), Rows, RowVolume, In.NumElements());
			return false;
		}
		EPyDType Want = EPyDType::Float32;
		FPyTensor::ParseDType(Spec.GetStringField(TEXT("dtype")), Want);
		if (In.DType == Want)
		{
			Out = In;
		}
		else if (Want == EPyDType::Float32)
		{
			Out = FPyTensor::FromFloats(In.ToFloats());
		}
		else
		{
			OutError = FString::Printf(TEXT("expected %s, got %s"), FPyTensor::DTypeName(Want), FPyTensor::DTypeName(In.DType));
			return false;
		}
		Out.Shape = { Rows };
		Out.Shape.Append(RowShape);
		return true;
	}

	//An image: (N, H, W, C) channels last (Unreal's) or (N, C, H, W), uint8 or float -> what the method's ut.Image makes
	TArray<int64> S = In.Shape;
	if (S.Num() == 3)
	{
		S.Add(1); //Rows of one-channel images (N, H, W)
	}
	if (S.Num() != 4 || S[0] != Rows)
	{
		OutError = TEXT("expected images (N, H, W, C) or (N, C, H, W)");
		return false;
	}
	if (In.DType != EPyDType::UInt8 && In.DType != EPyDType::Float32)
	{
		OutError = TEXT("images must be uint8 or float32");
		return false;
	}
	const FString Hint = Layout.ToUpper();
	bool bLast;
	if (Hint == TEXT("BGRA") || Hint == TEXT("RGBA") || Hint == TEXT("BGR") || Hint == TEXT("RGB") || Hint == TEXT("R") || Hint == TEXT("NHWC"))
	{
		bLast = true;
	}
	else if (Hint == TEXT("NCHW"))
	{
		bLast = false;
	}
	else
	{
		const bool bLastIsChannels = S[3] == 1 || S[3] == 3 || S[3] == 4;
		const bool bFirstIsChannels = S[1] == 1 || S[1] == 3 || S[1] == 4;
		bLast = bLastIsChannels && (!bFirstIsChannels || In.DType == EPyDType::UInt8);
	}
	const int64 N = S[0];
	const int32 C = (int32)(bLast ? S[3] : S[1]);
	const int32 H = (int32)(bLast ? S[1] : S[2]);
	const int32 W = (int32)(bLast ? S[2] : S[3]);
	const bool bBgr = Hint.StartsWith(TEXT("BGR"));
	bool bGray = false, bAlpha = false;
	Spec.TryGetBoolField(TEXT("gray"), bGray);
	Spec.TryGetBoolField(TEXT("alpha"), bAlpha);
	const FString Normalize = Spec.GetStringField(TEXT("normalize"));
	const bool bOutLast = Spec.GetStringField(TEXT("layout")).ToUpper() == TEXT("NHWC");
	const TArray<int64> Size = ShapeOf(Spec, TEXT("size"));
	const int32 OutH = Size.Num() == 2 ? (int32)Size[0] : H;
	const int32 OutW = Size.Num() == 2 ? (int32)Size[1] : W;
	const int32 OutC = bGray ? 1 : ((bAlpha && C == 4) ? 4 : 3);

	const uint8* Bytes = In.Data.GetData();
	const float* Floats = reinterpret_cast<const float*>(In.Data.GetData());
	auto Pixel = [&](int64 n, int32 c, int32 y, int32 x) -> float
	{
		const int64 i = bLast ? ((n * H + y) * W + x) * C + c : ((n * C + c) * H + y) * (int64)W + x;
		return In.DType == EPyDType::UInt8 ? Bytes[i] / 255.f : Floats[i];
	};
	const TArray<FTaps> RowTaps = ResizeWeights(H, OutH);
	const TArray<FTaps> ColTaps = ResizeWeights(W, OutW);
	static const float Mean[3] = { 0.485f, 0.456f, 0.406f };
	static const float Std[3] = { 0.229f, 0.224f, 0.225f };

	TArray<float> Result;
	Result.SetNumUninitialized(N * OutC * OutH * OutW);
	TArray<float> Plane, Resized;
	Plane.SetNumUninitialized(H * W);
	Resized.SetNumUninitialized(OutH * OutW);
	for (int64 n = 0; n < N; ++n)
	{
		for (int32 oc = 0; oc < OutC; ++oc)
		{
			//The channel: gray (mean of RGB), R / G / B (in the source's order) or alpha
			for (int32 y = 0; y < H; ++y)
			{
				for (int32 x = 0; x < W; ++x)
				{
					float V;
					if (C < 3)
					{
						V = Pixel(n, 0, y, x);
					}
					else if (bGray)
					{
						V = (Pixel(n, 0, y, x) + Pixel(n, 1, y, x) + Pixel(n, 2, y, x)) / 3.f;
					}
					else if (oc == 3)
					{
						V = Pixel(n, 3, y, x);
					}
					else
					{
						V = Pixel(n, bBgr ? 2 - oc : oc, y, x);
					}
					Plane[y * W + x] = V;
				}
			}
			const float* Src = Plane.GetData();
			if (OutH != H || OutW != W)
			{
				Resize(Plane.GetData(), H, W, Resized.GetData(), OutH, OutW, RowTaps, ColTaps);
				Src = Resized.GetData();
			}
			for (int32 y = 0; y < OutH; ++y)
			{
				for (int32 x = 0; x < OutW; ++x)
				{
					float V = Src[y * OutW + x];
					if (Normalize == TEXT("symmetric"))
					{
						V = V * 2.f - 1.f;
					}
					else if (Normalize == TEXT("imagenet") && oc < 3)
					{
						const float M = OutC == 1 ? (Mean[0] + Mean[1] + Mean[2]) / 3.f : Mean[oc];
						const float D = OutC == 1 ? (Std[0] + Std[1] + Std[2]) / 3.f : Std[oc];
						V = (V - M) / D;
					}
					const int64 o = bOutLast ? ((n * OutH + y) * OutW + x) * OutC + oc : ((n * OutC + oc) * OutH + y) * (int64)OutW + x;
					Result[o] = V;
				}
			}
		}
	}
	Out = FPyTensor::FromFloats(Result);
	Out.Shape = bOutLast ? TArray<int64>{ N, OutH, OutW, OutC } : TArray<int64>{ N, OutC, OutH, OutW };
	return true;
}

// ------------------------------------------------------------------------------------------- safetensors

bool FTorchNNE::ReadSafetensors(const FString& Path, TMap<FString, FPyTensor>& Out, FString& OutError)
{
	TArray64<uint8> File;
	if (!FFileHelper::LoadFileToArray(File, *Path) || File.Num() < 8)
	{
		OutError = TEXT("can't read ") + Path;
		return false;
	}
	uint64 HeaderSize = 0;
	FMemory::Memcpy(&HeaderSize, File.GetData(), 8);
	if (8 + HeaderSize > (uint64)File.Num())
	{
		OutError = TEXT("not a safetensors file: ") + Path;
		return false;
	}
	const FString HeaderText = FString(FUTF8ToTCHAR(reinterpret_cast<const ANSICHAR*>(File.GetData() + 8), (int32)HeaderSize));
	TSharedPtr<FJsonObject> Header;
	if (!FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(HeaderText), Header) || !Header.IsValid())
	{
		OutError = TEXT("bad safetensors header: ") + Path;
		return false;
	}
	static const TMap<FString, EPyDType> Types = {
		{ TEXT("F32"), EPyDType::Float32 }, { TEXT("F64"), EPyDType::Float64 }, { TEXT("F16"), EPyDType::Float16 }, { TEXT("I8"), EPyDType::Int8 },
		{ TEXT("U8"), EPyDType::UInt8 }, { TEXT("I16"), EPyDType::Int16 }, { TEXT("U16"), EPyDType::UInt16 }, { TEXT("I32"), EPyDType::Int32 },
		{ TEXT("U32"), EPyDType::UInt32 }, { TEXT("I64"), EPyDType::Int64 }, { TEXT("U64"), EPyDType::UInt64 }, { TEXT("BOOL"), EPyDType::Bool } };
	const uint8* Data = File.GetData() + 8 + HeaderSize;
	for (const TPair<FString, TSharedPtr<FJsonValue>>& Field : Header->Values)
	{
		if (Field.Key == TEXT("__metadata__"))
		{
			continue;
		}
		const TSharedPtr<FJsonObject> Desc = Field.Value->AsObject();
		const EPyDType* Type = Desc.IsValid() ? Types.Find(Desc->GetStringField(TEXT("dtype"))) : nullptr;
		const TArray<TSharedPtr<FJsonValue>>* Offsets = nullptr;
		if (!Type || !Desc->TryGetArrayField(TEXT("data_offsets"), Offsets) || Offsets->Num() != 2)
		{
			OutError = FString::Printf(TEXT("%s: unsupported tensor '%s'"), *Path, *Field.Key);
			return false;
		}
		FPyTensor T;
		T.DType = *Type;
		T.Shape = TorchNNEPrivate::ShapeOf(*Desc, TEXT("shape"));
		const int64 Begin = (int64)(*Offsets)[0]->AsNumber();
		const int64 End = (int64)(*Offsets)[1]->AsNumber();
		T.Data.Append(Data + Begin, End - Begin);
		Out.Add(Field.Key, MoveTemp(T));
	}
	return true;
}
