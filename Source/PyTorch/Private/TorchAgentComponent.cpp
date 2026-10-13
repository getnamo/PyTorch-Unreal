#include "TorchAgentComponent.h"
#include "PyTorch.h"
#include "TorchEnvironment.h"
#include "EngineUtils.h"
#include "Engine/World.h"

UTorchAgentComponent::UTorchAgentComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
}

void UTorchAgentComponent::BeginPlay()
{
	Super::BeginPlay();
	if (!Environment && GetWorld())
	{
		for (TActorIterator<ATorchEnvironment> It(GetWorld()); It; ++It)
		{
			Environment = *It;
			break;
		}
	}
	if (Environment)
	{
		Environment->RegisterAgent(this);
	}
	else
	{
		UE_LOG(LogTorch, Warning, TEXT("%s: no Torch Environment in the world"), *GetPathName());
	}
}

void UTorchAgentComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	if (Environment)
	{
		Environment->UnregisterAgent(this);
	}
	Super::EndPlay(EndPlayReason);
}

void UTorchAgentComponent::Init(ATorchEnvironment* InEnvironment, int32 ObservationSize, const TMap<FString, TPair<int32, int32>>* InLayout)
{
	Environment = InEnvironment;
	Layout = InLayout;
	Observation.SetNumZeroed(ObservationSize);
}

void UTorchAgentComponent::SetObservation(const FString& Name, float Value)
{
	SetObservations(Name, { Value });
}

void UTorchAgentComponent::SetObservationVector(const FString& Name, FVector Value)
{
	if (Layout && !Layout->Contains(Name) && Layout->Contains(Name + TEXT(".x")))
	{
		SetObservation(Name + TEXT(".x"), (float)Value.X);
		SetObservation(Name + TEXT(".y"), (float)Value.Y);
		SetObservation(Name + TEXT(".z"), (float)Value.Z);
		return;
	}
	SetObservations(Name, { (float)Value.X, (float)Value.Y, (float)Value.Z });
}

void UTorchAgentComponent::SetObservations(const FString& Name, const TArray<float>& Values)
{
	const TPair<int32, int32>* Slot = Layout ? Layout->Find(Name) : nullptr;
	if (!Slot)
	{
		if (!Warned.Contains(Name))
		{
			Warned.Add(Name);
			UE_LOG(LogTorch, Warning, TEXT("%s: observation '%s' isn't in the environment's spec"), *GetPathName(), *Name);
		}
		return;
	}
	const int32 Count = FMath::Min(Slot->Value, Values.Num());
	for (int32 i = 0; i < Count; ++i)
	{
		Observation[Slot->Key + i] = Values[i];
	}
}

void UTorchAgentComponent::AddReward(float InReward)
{
	Reward += InReward;
	EpisodeReturn += InReward;
}

void UTorchAgentComponent::SetReward(float InReward)
{
	EpisodeReturn += InReward - Reward;
	Reward = InReward;
}

void UTorchAgentComponent::EndEpisode(bool bInTerminated)
{
	bDone = true;
	bTerminated = bInTerminated;
}

void UTorchAgentComponent::SetHeuristicActions(const TArray<float>& InActions)
{
	HeuristicActions = InActions;
}

float UTorchAgentComponent::RandomRange(float Min, float Max)
{
	return GetRandomStream().FRandRange(Min, Max);
}

FRandomStream& UTorchAgentComponent::GetRandomStream() const
{
	static FRandomStream Fallback(0);
	return Environment ? Environment->GetRandomStream() : Fallback;
}

float UTorchAgentComponent::GetStepSeconds() const
{
	return Environment ? Environment->StepSeconds : 1.f / 30.f;
}

float UTorchAgentComponent::ConsumeReward()
{
	const float R = Reward;
	Reward = 0.f;
	return R;
}

void UTorchAgentComponent::BeginEpisode()
{
	Reward = 0.f;
	EpisodeReturn = 0.f;
	EpisodeSteps = 0;
	bDone = false;
	bTerminated = false;
}
