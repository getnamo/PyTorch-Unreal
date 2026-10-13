#include "TorchEnvironment.h"
#include "PyTorch.h"
#include "PythonJob.h"
#include "PythonProcess.h"
#include "PythonService.h"
#include "PythonStream.h"
#include "TorchAgentComponent.h"
#include "TorchBlueprintLibrary.h"
#include "TorchModel.h"
#include "TorchSubsystem.h"
#include "Components/SceneComponent.h"
#include "Dom/JsonObject.h"
#include "HAL/FileManager.h"
#include "Misc/DateTime.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"

namespace TorchEnvironmentPrivate
{
	/** A safetensors file: an 8-byte header size, a JSON header, the raw tensors (little-endian, row-major). */
	struct FSafetensorsWriter
	{
		TSharedRef<FJsonObject> Header = MakeShared<FJsonObject>();
		TArray<uint8> Data;

		void Add(const FString& Name, const TCHAR* DType, TArray<int64> Shape, const void* Bytes, int64 NumBytes)
		{
			TSharedRef<FJsonObject> T = MakeShared<FJsonObject>();
			T->SetStringField(TEXT("dtype"), DType);
			TArray<TSharedPtr<FJsonValue>> S;
			for (int64 D : Shape)
			{
				S.Add(MakeShared<FJsonValueNumber>((double)D));
			}
			T->SetArrayField(TEXT("shape"), S);
			T->SetArrayField(TEXT("data_offsets"), { MakeShared<FJsonValueNumber>((double)Data.Num()), MakeShared<FJsonValueNumber>((double)(Data.Num() + NumBytes)) });
			Header->SetObjectField(Name, T);
			Data.Append(static_cast<const uint8*>(Bytes), NumBytes);
		}

		bool Save(const FString& Path, const TSharedRef<FJsonObject>& Metadata)
		{
			TSharedRef<FJsonObject> Meta = MakeShared<FJsonObject>();
			for (const TPair<FString, TSharedPtr<FJsonValue>>& Field : Metadata->Values)
			{
				Meta->SetStringField(Field.Key, Field.Value->Type == EJson::String ? Field.Value->AsString() : PythonWire::ToJsonString(Field.Value));
			}
			Header->SetObjectField(TEXT("__metadata__"), Meta);
			FTCHARToUTF8 Json(*PythonWire::ToJsonString(MakeShared<FJsonValueObject>(Header)));
			int64 HeaderSize = Json.Length();
			const int64 Padded = (HeaderSize + 7) / 8 * 8;
			TArray<uint8> Out;
			Out.Reserve(8 + Padded + Data.Num());
			Out.Append(reinterpret_cast<const uint8*>(&Padded), 8);
			Out.Append(reinterpret_cast<const uint8*>(Json.Get()), HeaderSize);
			for (int64 i = HeaderSize; i < Padded; ++i)
			{
				Out.Add(' ');
			}
			Out.Append(Data);
			return FFileHelper::SaveArrayToFile(Out, *Path);
		}
	};
}

ATorchEnvironment::ATorchEnvironment()
{
	PrimaryActorTick.bCanEverTick = true;
	PrimaryActorTick.TickGroup = TG_PrePhysics;
	RootComponent = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
}

int32 ATorchEnvironment::GetObservationSize() const
{
	int32 Size = 0;
	for (const FTorchObservationSpec& O : Observations)
	{
		Size += FMath::Max(1, O.Size);
	}
	return Size;
}

void ATorchEnvironment::BuildLayout()
{
	Layout.Reset();
	int32 Offset = 0;
	for (const FTorchObservationSpec& O : Observations)
	{
		Layout.Add(O.Name, TPair<int32, int32>(Offset, FMath::Max(1, O.Size)));
		Offset += FMath::Max(1, O.Size);
	}
}

void ATorchEnvironment::RegisterAgent(UTorchAgentComponent* Agent)
{
	if (!Agent || Agents.Contains(Agent))
	{
		return;
	}
	if (Layout.Num() == 0)
	{
		BuildLayout();
	}
	Agent->Init(this, GetObservationSize(), &Layout);
	Agents.Add(Agent);
	AgentRefs.Add(Agent);
	bNeedsReset = true; //A new sub-environment: everyone starts a fresh episode (Gymnasium's vector size is fixed)
	if (TrainStream.IsValid())
	{
		CloseTrainStream(); //The frame size changed: Python sees a new environment
	}
}

void ATorchEnvironment::UnregisterAgent(UTorchAgentComponent* Agent)
{
	Agents.Remove(Agent);
	AgentRefs.Remove(Agent);
}

void ATorchEnvironment::BeginPlay()
{
	BuildLayout();
	Random.Initialize(Seed);
	Super::BeginPlay();
	if (Mode == ETorchEnvironmentMode::Train && bStartTrainingOnBeginPlay)
	{
		StartTraining();
	}
}

void ATorchEnvironment::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	FlushRecording();
	CloseTrainStream();
	Super::EndPlay(EndPlayReason);
}

void ATorchEnvironment::Tick(float DeltaSeconds)
{
	Super::Tick(DeltaSeconds);
	for (int32 i = 0; i < StepsPerTick; ++i)
	{
		if (!StepEnvironment())
		{
			break;
		}
	}
}

void ATorchEnvironment::SetMode(ETorchEnvironmentMode NewMode)
{
	if (NewMode == Mode)
	{
		return;
	}
	if (Mode == ETorchEnvironmentMode::Train)
	{
		CloseTrainStream();
	}
	if (Mode == ETorchEnvironmentMode::Record)
	{
		FlushRecording();
		Recording.PendingObs.Reset();
		Recording.PendingActions.Reset();
		Recording.Directory.Reset(); //The next recording is a new session
		Recording.Shard = 0;
	}
	Mode = NewMode;
	bNeedsReset = true;
	bRunPending = false;
	bRunActionsReady = false;
	++RunRequestGeneration;
	if (Mode == ETorchEnvironmentMode::Train && HasActorBegunPlay() && (!TrainingJobHandle || TrainingJobHandle->IsFinished()))
	{
		StartTraining();
	}
}

void ATorchEnvironment::ResetAllAgents()
{
	bNeedsReset = true;
}

void ATorchEnvironment::ResetWithSeed(int32 NewSeed)
{
	Seed = NewSeed;
	Random.Initialize(Seed);
	bNeedsReset = true;
	bRunPending = false; //A policy request on its way answers the old episode: dropped
	bRunActionsReady = false;
	++RunRequestGeneration;
}

void ATorchEnvironment::GetStats(int64& OutSteps, int32& OutEpisodes, float& OutMeanReturn) const
{
	OutSteps = TotalSteps;
	OutEpisodes = Episodes;
	float Sum = 0.f;
	for (float R : RecentReturns)
	{
		Sum += R;
	}
	OutMeanReturn = RecentReturns.Num() ? Sum / RecentReturns.Num() : 0.f;
}

void ATorchEnvironment::Collect(TArray<float>& Obs, TArray<float>& FinalObs, TArray<float>& Rewards, TArray<float>& Terminated, TArray<float>& Truncated, bool bResetAll)
{
	const int32 D = GetObservationSize();
	const int32 N = Agents.Num();
	Obs.SetNumZeroed(N * D);
	FinalObs.SetNumZeroed(N * D);
	Rewards.SetNumZeroed(N);
	Terminated.SetNumZeroed(N);
	Truncated.SetNumZeroed(N);
	for (int32 i = 0; i < N; ++i)
	{
		UTorchAgentComponent* Agent = Agents[i];
		if (bResetAll)
		{
			Agent->BeginEpisode();
			Agent->ResetAgent();
			Agent->CollectObservations();
			FMemory::Memcpy(&Obs[i * D], Agent->GetObservationBuffer().GetData(), D * sizeof(float));
			continue;
		}
		Agent->CollectObservations();
		Rewards[i] = Agent->ConsumeReward();
		const bool bTimeUp = MaxEpisodeSteps > 0 && Agent->GetEpisodeSteps() >= MaxEpisodeSteps;
		if (Agent->IsDone() || bTimeUp)
		{
			const bool bTerm = Agent->IsDone() && Agent->IsTerminated();
			Terminated[i] = bTerm ? 1.f : 0.f;
			Truncated[i] = bTerm ? 0.f : 1.f;
			FMemory::Memcpy(&FinalObs[i * D], Agent->GetObservationBuffer().GetData(), D * sizeof(float));
			++Episodes;
			RecentReturns.Add(Agent->GetEpisodeReturn());
			if (RecentReturns.Num() > 100)
			{
				RecentReturns.RemoveAt(0);
			}
			OnEpisodeEnded.Broadcast(Agent, Agent->GetEpisodeReturn(), bTerm);
			//Gymnasium SAME_STEP autoreset: the new episode's first observation goes out in this same step
			Agent->BeginEpisode();
			Agent->ResetAgent();
			Agent->CollectObservations();
		}
		FMemory::Memcpy(&Obs[i * D], Agent->GetObservationBuffer().GetData(), D * sizeof(float));
	}
}

void ATorchEnvironment::Apply(const TArray<float>& Flat)
{
	const int32 A = Actions.Width();
	TArray<float> One;
	One.SetNumUninitialized(A);
	for (int32 i = 0; i < Agents.Num(); ++i)
	{
		for (int32 k = 0; k < A; ++k)
		{
			One[k] = Flat.IsValidIndex(i * A + k) ? Flat[i * A + k] : 0.f;
		}
		Agents[i]->SetActions(One);
		Agents[i]->CountStep();
		Agents[i]->ApplyActions(One);
	}
}

bool ATorchEnvironment::StepEnvironment()
{
	Agents.RemoveAll([](UTorchAgentComponent* A) { return !IsValid(A); });
	if (Agents.Num() == 0)
	{
		return false;
	}
	switch (Mode)
	{
	case ETorchEnvironmentMode::Train: return StepTrain();
	case ETorchEnvironmentMode::Run: return StepRun();
	case ETorchEnvironmentMode::Heuristic: return StepHeuristic(false);
	case ETorchEnvironmentMode::Record: return StepHeuristic(true);
	default: return false;
	}
}

// ------------------------------------------------------------------------------------------------- training

UPythonJob* ATorchEnvironment::StartTraining()
{
	UTorchSubsystem* Torch = UTorchSubsystem::Get();
	if (!Torch || !TrainingModel)
	{
		UE_LOG(LogTorch, Warning, TEXT("%s: Train mode needs a Training Model (the Torch Model whose script has the job)"), *GetName());
		return nullptr;
	}
	if (TrainingJobHandle && !TrainingJobHandle->IsFinished())
	{
		return TrainingJobHandle;
	}
	CloseTrainStream();
	TrainingJobHandle = Torch->StartJob(TrainingModel, TrainingJob, TrainingParamsJson, false, bTrainInOwnProcess);
	bNeedsReset = true;
	return TrainingJobHandle;
}

void ATorchEnvironment::StopTraining()
{
	if (TrainingJobHandle)
	{
		TrainingJobHandle->Stop();
	}
	CloseTrainStream();
}

bool ATorchEnvironment::OpenTrainStream()
{
	UPythonService* Service = TrainingJobHandle ? TrainingJobHandle->GetService() : nullptr;
	if (!Service || TrainingJobHandle->IsFinished())
	{
		return false;
	}
	if (TrainingJobHandle->State != EPythonJobState::Running)
	{
		return false; //Not started yet (the process is coming up): don't block the game thread waiting for it
	}
	const TSharedPtr<FPythonProcess, ESPMode::ThreadSafe> Process = Service->GetProcess();
	if (!Process.IsValid() || !Process->GetChannel()->IsRunning())
	{
		return false;
	}
	const int32 D = GetObservationSize();
	TSharedRef<FJsonObject> Info = MakeShared<FJsonObject>();
	Info->SetStringField(TEXT("call"), TEXT("_torch_env"));
	Info->SetStringField(TEXT("env"), EnvName);
	Info->SetStringField(TEXT("name"), EnvName);
	Info->SetNumberField(TEXT("obs_size"), D);
	TArray<TSharedPtr<FJsonValue>> Names;
	for (const FTorchObservationSpec& O : Observations)
	{
		for (int32 k = 0; k < FMath::Max(1, O.Size); ++k)
		{
			Names.Add(MakeShared<FJsonValueString>(O.Size > 1 ? FString::Printf(TEXT("%s.%d"), *O.Name, k) : O.Name));
		}
	}
	Info->SetArrayField(TEXT("obs_names"), Names);
	TSharedRef<FJsonObject> Action = MakeShared<FJsonObject>();
	Action->SetStringField(TEXT("type"), Actions.Type == ETorchActionType::Discrete ? TEXT("discrete") : TEXT("continuous"));
	Action->SetNumberField(TEXT("size"), Actions.Size);
	Action->SetNumberField(TEXT("n"), Actions.Choices);
	TArray<TSharedPtr<FJsonValue>> Low, High;
	for (int32 k = 0; k < Actions.Size; ++k)
	{
		Low.Add(MakeShared<FJsonValueNumber>(Actions.Low));
		High.Add(MakeShared<FJsonValueNumber>(Actions.High));
	}
	Action->SetArrayField(TEXT("low"), Low);
	Action->SetArrayField(TEXT("high"), High);
	Info->SetObjectField(TEXT("action"), Action);

	FPythonStreamConfig Config;
	Config.Owner = Service->GetOwnerId();
	Config.Script = Service->ScriptParams.Script;
	Config.Name = FString::Printf(TEXT("env:%s#%u"), *EnvName, GetUniqueID());
	Config.Policy = EPythonStreamPolicy::Lockstep;
	Config.Slots = 1;
	Config.SlotBytes = (int64)Agents.Num() * (2 * D + 3) * sizeof(float);
	Config.Info = Info;
	FString Error;
	TrainStream = FPythonStream::Open(Process.ToSharedRef(), Config, Error);
	if (!TrainStream.IsValid())
	{
		UE_LOG(LogTorch, Error, TEXT("%s: the training stream couldn't open: %s"), *GetName(), *Error);
		return false;
	}
	bNeedsReset = true;
	return true;
}

void ATorchEnvironment::CloseTrainStream()
{
	if (TrainStream.IsValid())
	{
		TrainStream->Close();
		TrainStream.Reset();
	}
}

void ATorchEnvironment::EndTraining(const TCHAR* Why)
{
	UE_LOG(LogTorch, Log, TEXT("%s: training ended (%s)"), *GetName(), Why);
	CloseTrainStream();
	SetMode(ModeAfterTraining == ETorchEnvironmentMode::Run && !PolicyModel ? ETorchEnvironmentMode::Off
		: ModeAfterTraining == ETorchEnvironmentMode::Train ? ETorchEnvironmentMode::Off : ModeAfterTraining);
}

bool ATorchEnvironment::StepTrain()
{
	if (TrainingJobHandle && TrainingJobHandle->IsFinished())
	{
		EndTraining(TEXT("the job finished"));
		return false;
	}
	if (!TrainStream.IsValid() && !OpenTrainStream())
	{
		return false;
	}
	const int32 D = GetObservationSize();
	const int32 N = Agents.Num();
	TArray<float> Obs, FinalObs, Rewards, Term, Trunc;
	const bool bReset = bNeedsReset;
	Collect(Obs, FinalObs, Rewards, Term, Trunc, bReset);
	bNeedsReset = false;
	const int32 W = 2 * D + 3;
	TArray<float> Rows;
	Rows.SetNumUninitialized(N * W);
	for (int32 i = 0; i < N; ++i)
	{
		float* R = &Rows[i * W];
		FMemory::Memcpy(R, &Obs[i * D], D * sizeof(float));
		FMemory::Memcpy(R + D, &FinalObs[i * D], D * sizeof(float));
		R[2 * D] = Rewards[i];
		R[2 * D + 1] = Term[i];
		R[2 * D + 2] = Trunc[i];
	}
	FPythonStreamFrame Frame;
	Frame.DType = EPyDType::Float32;
	Frame.Shape = { N, W };
	Frame.NumBytes = (int64)Rows.Num() * sizeof(float);
	Frame.Meta = MakeShared<FJsonObject>();
	Frame.Meta->SetBoolField(TEXT("reset"), bReset);
	Frame.Meta->SetNumberField(TEXT("step"), (double)TotalSteps);
	FPythonCallResult Result;
	if (!TrainStream->Step(Rows.GetData(), Frame, StepTimeoutSeconds, Result))
	{
		UE_LOG(LogTorch, Warning, TEXT("%s: training step failed (%s): leaving Train mode"), *GetName(), *Result.Error);
		EndTraining(TEXT("the step failed"));
		return false;
	}
	const TSharedPtr<FJsonObject>* Reply = nullptr;
	if (!Result.Result.IsValid() || !Result.Result->TryGetObject(Reply))
	{
		return true;
	}
	bool bFlag = false;
	double SeedValue = 0.0;
	if ((*Reply)->TryGetNumberField(TEXT("seed"), SeedValue))
	{
		Random.Initialize((int32)SeedValue);
	}
	if ((*Reply)->TryGetBoolField(TEXT("stop"), bFlag) && bFlag)
	{
		EndTraining(TEXT("the trainer closed its env"));
		return false;
	}
	if ((*Reply)->TryGetBoolField(TEXT("reset"), bFlag) && bFlag)
	{
		bNeedsReset = true;
		return true;
	}
	const TSharedPtr<FJsonObject>* Desc = nullptr;
	double Index = -1.0;
	if ((*Reply)->TryGetObjectField(TEXT("actions"), Desc) && (*Desc)->TryGetNumberField(TEXT("_tensor"), Index) && Result.Tensors.IsValidIndex((int32)Index))
	{
		Apply(Result.Tensors[(int32)Index].ToFloats());
		++TotalSteps;
	}
	return true;
}

// ---------------------------------------------------------------------------------------------------- run

bool ATorchEnvironment::StepRun()
{
	UTorchSubsystem* Torch = UTorchSubsystem::Get();
	if (!Torch || !PolicyModel || bRunPending)
	{
		return false; //Turn-based: no step until this one's actions are in
	}
	if (bRunActionsReady)
	{
		Apply(RunActions);
		bRunActionsReady = false;
		++TotalSteps;
	}
	const int32 D = GetObservationSize();
	TArray<float> Obs, FinalObs, Rewards, Term, Trunc;
	Collect(Obs, FinalObs, Rewards, Term, Trunc, bNeedsReset);
	bNeedsReset = false;
	TMap<FString, FPyTensor> Inputs;
	Inputs.Add(PolicyInput, FPyTensor::FromFloats(Obs, { Agents.Num(), D }));
	FTorchRunOptions Options;
	Options.bBatched = true;
	Options.bImmediate = true;
	bRunPending = true;
	TWeakObjectPtr<ATorchEnvironment> Weak(this);
	const int32 Generation = RunRequestGeneration;
	Torch->Run(PolicyModel, PolicyMethod, Inputs, FString(), [Weak, Generation](const FTorchResult& R)
	{
		ATorchEnvironment* Self = Weak.Get();
		if (!Self || Self->RunRequestGeneration != Generation)
		{
			return;
		}
		Self->bRunPending = false;
		if (!R.bSuccess)
		{
			UE_LOG(LogTorch, Warning, TEXT("%s: the policy failed (%s): leaving Run mode"), *Self->GetName(), *R.Error);
			Self->Mode = ETorchEnvironmentMode::Off;
			return;
		}
		const FPyTensor* Out = R.Outputs.Find(TEXT("actions"));
		if (!Out)
		{
			Out = R.Outputs.Find(TEXT("output"));
		}
		Self->RunActions = Out ? Out->ToFloats() : TArray<float>();
		if (Self->bAgentsMoveWithWorld)
		{
			//Now, before the world ticks on: the next step collects what these actions did
			Self->Apply(Self->RunActions);
			++Self->TotalSteps;
			return;
		}
		Self->bRunActionsReady = true;
	}, Options);
	return true;
}

// --------------------------------------------------------------------------------------- heuristic, record

bool ATorchEnvironment::StepHeuristic(bool bRecord)
{
	TArray<float> Obs, FinalObs, Rewards, Term, Trunc;
	const bool bReset = bNeedsReset;
	Collect(Obs, FinalObs, Rewards, Term, Trunc, bReset);
	bNeedsReset = false;
	if (bRecord && !bReset)
	{
		Record(Obs, FinalObs, Rewards, Term, Trunc);
	}
	const int32 A = Actions.Width();
	TArray<float> Flat;
	Flat.SetNumZeroed(Agents.Num() * A);
	for (int32 i = 0; i < Agents.Num(); ++i)
	{
		Agents[i]->Heuristic();
		const TArray<float>& H = Agents[i]->GetHeuristicActions();
		for (int32 k = 0; k < A && k < H.Num(); ++k)
		{
			Flat[i * A + k] = H[k];
		}
	}
	if (bRecord)
	{
		Recording.PendingObs = Obs;
		Recording.PendingActions = Flat;
	}
	Apply(Flat);
	++TotalSteps;
	return true;
}

void ATorchEnvironment::Record(const TArray<float>& Obs, const TArray<float>& FinalObs, const TArray<float>& Rewards, const TArray<float>& Terminated, const TArray<float>& Truncated)
{
	const int32 D = GetObservationSize();
	const int32 N = Rewards.Num();
	if (Recording.PendingObs.Num() != N * D || Recording.PendingActions.Num() != N * Actions.Width())
	{
		return; //Nothing pending (the first step) or the agents changed
	}
	if (Recording.Directory.IsEmpty())
	{
		Recording.Directory = FPaths::ConvertRelativePathToFull(FPaths::ProjectSavedDir() / TEXT("Torch/Recordings") / RecordingName / FDateTime::Now().ToString(TEXT("%Y%m%d-%H%M%S")));
		IFileManager::Get().MakeDirectory(*Recording.Directory, true);
	}
	Recording.Obs.Append(Recording.PendingObs);
	Recording.Actions.Append(Recording.PendingActions);
	Recording.Rewards.Append(Rewards);
	Recording.Terminated.Append(Terminated);
	Recording.Truncated.Append(Truncated);
	for (int32 i = 0; i < N; ++i)
	{
		const bool bEnded = Terminated[i] > 0.5f || Truncated[i] > 0.5f;
		Recording.NextObs.Append(bEnded ? &FinalObs[i * D] : &Obs[i * D], D);
		Recording.AgentIds.Add(i);
	}
	Recording.PendingObs.Reset();
	Recording.PendingActions.Reset();
	if (++Recording.Steps >= StepsPerShard)
	{
		FlushRecording();
	}
}

void ATorchEnvironment::FlushRecording()
{
	using namespace TorchEnvironmentPrivate;
	if (Recording.Rewards.Num() == 0)
	{
		return;
	}
	const int64 Rows = Recording.Rewards.Num();
	const int32 D = GetObservationSize();
	const int32 A = Actions.Width();
	FSafetensorsWriter W;
	W.Add(TEXT("obs"), TEXT("F32"), { Rows, D }, Recording.Obs.GetData(), Recording.Obs.Num() * sizeof(float));
	W.Add(TEXT("actions"), TEXT("F32"), { Rows, A }, Recording.Actions.GetData(), Recording.Actions.Num() * sizeof(float));
	W.Add(TEXT("rewards"), TEXT("F32"), { Rows }, Recording.Rewards.GetData(), Rows * sizeof(float));
	W.Add(TEXT("terminated"), TEXT("F32"), { Rows }, Recording.Terminated.GetData(), Rows * sizeof(float));
	W.Add(TEXT("truncated"), TEXT("F32"), { Rows }, Recording.Truncated.GetData(), Rows * sizeof(float));
	W.Add(TEXT("next_obs"), TEXT("F32"), { Rows, D }, Recording.NextObs.GetData(), Recording.NextObs.Num() * sizeof(float));
	W.Add(TEXT("agent"), TEXT("I32"), { Rows }, Recording.AgentIds.GetData(), Rows * sizeof(int32));
	TSharedRef<FJsonObject> Meta = MakeShared<FJsonObject>();
	Meta->SetStringField(TEXT("env"), EnvName);
	Meta->SetNumberField(TEXT("obs_size"), D);
	Meta->SetNumberField(TEXT("action_size"), A);
	Meta->SetStringField(TEXT("action_type"), Actions.Type == ETorchActionType::Discrete ? TEXT("discrete") : TEXT("continuous"));
	Meta->SetNumberField(TEXT("agents"), Agents.Num());
	Meta->SetNumberField(TEXT("step_seconds"), StepSeconds);
	const FString Path = Recording.Directory / FString::Printf(TEXT("shard_%05d.safetensors"), Recording.Shard++);
	if (W.Save(Path, Meta))
	{
		UE_LOG(LogTorch, Log, TEXT("%s: recorded %lld rows to %s"), *GetName(), Rows, *Path);
	}
	else
	{
		UE_LOG(LogTorch, Error, TEXT("%s: couldn't write %s"), *GetName(), *Path);
	}
	Recording.Obs.Reset();
	Recording.Actions.Reset();
	Recording.Rewards.Reset();
	Recording.Terminated.Reset();
	Recording.Truncated.Reset();
	Recording.NextObs.Reset();
	Recording.AgentIds.Reset();
	Recording.Steps = 0;
}
