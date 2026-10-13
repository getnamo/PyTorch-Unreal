#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Math/RandomStream.h"
#include "TorchAgentComponent.generated.h"

class ATorchEnvironment;
class UTorchAgentComponent;

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FTorchAgentSignature, UTorchAgentComponent*, Agent);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FTorchAgentActionsSignature, UTorchAgentComponent*, Agent, const TArray<float>&, Actions);

/**
 * One agent of a Torch Environment (reinforcement learning). Each decision step the environment asks it for its
 * observations (On Collect Observations: Set Observation ...), gives it the policy's actions (On Apply Actions),
 * and resets it when its episode ends (On Reset Agent). The agent reports rewards (Add Reward) and ends its
 * episode (End Episode). C++ agents can override the Collect / Apply / Reset / Heuristic functions instead.
 */
UCLASS(ClassGroup = "Torch", meta = (BlueprintSpawnableComponent))
class PYTORCH_API UTorchAgentComponent : public UActorComponent
{
	GENERATED_BODY()
public:
	UTorchAgentComponent();

	/** The environment this agent belongs to. Unset: the first Torch Environment in the world. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Torch Agent")
	TObjectPtr<ATorchEnvironment> Environment;

	/** Fill the observations: Set Observation / Set Observation Vector / Set Observations. */
	UPROPERTY(BlueprintAssignable, Category = "Torch Agent")
	FTorchAgentSignature OnCollectObservations;

	/** The policy's actions for this step (continuous: the action values; discrete: one value, the choice). */
	UPROPERTY(BlueprintAssignable, Category = "Torch Agent")
	FTorchAgentActionsSignature OnApplyActions;

	/** Start a new episode (use Get Random Stream for reproducible starts). */
	UPROPERTY(BlueprintAssignable, Category = "Torch Agent")
	FTorchAgentSignature OnResetAgent;

	/** Heuristic and Record modes: choose the actions yourself (Set Heuristic Actions), e.g. from player input. */
	UPROPERTY(BlueprintAssignable, Category = "Torch Agent")
	FTorchAgentSignature OnHeuristic;

	UFUNCTION(BlueprintCallable, Category = "Torch Agent")
	void SetObservation(const FString& Name, float Value);

	/** Writes Name's first three values (its spec's Size must be 3), or Name.x / .y / .z if declared that way. */
	UFUNCTION(BlueprintCallable, Category = "Torch Agent")
	void SetObservationVector(const FString& Name, FVector Value);

	UFUNCTION(BlueprintCallable, Category = "Torch Agent")
	void SetObservations(const FString& Name, const TArray<float>& Values);

	UFUNCTION(BlueprintCallable, Category = "Torch Agent")
	void AddReward(float Reward);

	UFUNCTION(BlueprintCallable, Category = "Torch Agent")
	void SetReward(float Reward);

	/** End the episode now: Terminated (reached a goal or failed) or truncated (out of time, not the agent's
	 doing). The environment resets the agent at the end of this step. */
	UFUNCTION(BlueprintCallable, Category = "Torch Agent")
	void EndEpisode(bool bTerminated = true);

	UFUNCTION(BlueprintCallable, Category = "Torch Agent")
	void SetHeuristicActions(const TArray<float>& Actions);

	UFUNCTION(BlueprintPure, Category = "Torch Agent")
	const TArray<float>& GetLastActions() const { return Actions; }

	UFUNCTION(BlueprintPure, Category = "Torch Agent")
	int32 GetEpisodeSteps() const { return EpisodeSteps; }

	UFUNCTION(BlueprintPure, Category = "Torch Agent")
	float GetEpisodeReturn() const { return EpisodeReturn; }

	/** A random number from the environment's seeded stream: use it for resets, so runs replay exactly. */
	UFUNCTION(BlueprintCallable, Category = "Torch Agent")
	float RandomRange(float Min, float Max);

	/** C++: the environment's seeded random stream. */
	FRandomStream& GetRandomStream() const;

	/** The environment's simulated seconds per decision step (fixed: integrate with it, not with frame time). */
	UFUNCTION(BlueprintPure, Category = "Torch Agent")
	float GetStepSeconds() const;

	// -- C++ agents override these (the defaults broadcast the events above) ---------------------------------
	virtual void CollectObservations() { OnCollectObservations.Broadcast(this); }
	virtual void ApplyActions(const TArray<float>& InActions) { OnApplyActions.Broadcast(this, InActions); }
	virtual void ResetAgent() { OnResetAgent.Broadcast(this); }
	virtual void Heuristic() { OnHeuristic.Broadcast(this); }

	// -- The environment's side ---------------------------------------------------------------------------
	void Init(ATorchEnvironment* InEnvironment, int32 ObservationSize, const TMap<FString, TPair<int32, int32>>* InLayout);
	const TArray<float>& GetObservationBuffer() const { return Observation; }
	/** Take this step's reward (and reset the accumulator). */
	float ConsumeReward();
	bool IsDone() const { return bDone; }
	bool IsTerminated() const { return bTerminated; }
	void BeginEpisode();
	void CountStep() { ++EpisodeSteps; }
	void SetActions(const TArray<float>& InActions) { Actions = InActions; }
	const TArray<float>& GetHeuristicActions() const { return HeuristicActions; }

	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

private:
	TArray<float> Observation;
	TArray<float> Actions;
	TArray<float> HeuristicActions;
	/** Name -> (offset, size) in the observation vector; owned by the environment. */
	const TMap<FString, TPair<int32, int32>>* Layout = nullptr;
	float Reward = 0.f;
	float EpisodeReturn = 0.f;
	int32 EpisodeSteps = 0;
	bool bDone = false;
	bool bTerminated = false;
	TSet<FString> Warned;
};
