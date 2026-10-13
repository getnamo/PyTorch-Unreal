#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Math/RandomStream.h"
#include "PythonTypes.h"
#include "TorchTypes.h"
#include "TorchEnvironment.generated.h"

class FPythonStream;
class UPythonJob;
class UTorchAgentComponent;
class UTorchModel;

/** One named block of an agent's observation vector. */
USTRUCT(BlueprintType)
struct PYTORCH_API FTorchObservationSpec
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Torch")
	FString Name;

	/** Values in this block (a vector is 3). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Torch", meta = (ClampMin = "1"))
	int32 Size = 1;
};

UENUM(BlueprintType)
enum class ETorchActionType : uint8
{
	/** Size values in [Low, High] (Gymnasium Box). */
	Continuous,
	/** One choice out of Choices (Gymnasium Discrete). */
	Discrete,
};

USTRUCT(BlueprintType)
struct PYTORCH_API FTorchActionSpec
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Torch")
	ETorchActionType Type = ETorchActionType::Continuous;

	/** Continuous: how many values. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Torch", meta = (ClampMin = "1"))
	int32 Size = 1;

	/** Discrete: how many choices. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Torch", meta = (ClampMin = "2"))
	int32 Choices = 2;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Torch")
	float Low = -1.f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Torch")
	float High = 1.f;

	int32 Width() const { return Type == ETorchActionType::Discrete ? 1 : Size; }
};

UENUM(BlueprintType)
enum class ETorchEnvironmentMode : uint8
{
	/** Agents stay still. */
	Off,
	/** A Python training job steps the agents in lockstep (UnrealVectorEnv): the game waits for each step's
	 actions. */
	Train,
	/** The Policy Model acts (Python, or NNE in shipping builds). Turn-based: a step happens once its actions
	 are in, so a run replays exactly with the same seed. */
	Run,
	/** The agents' On Heuristic chooses the actions (e.g. player input or a scripted controller). */
	Heuristic,
	/** Heuristic, and every step is recorded to Saved/Torch/Recordings/<Recording Name>/ (ut.data.Recording):
	 one row per agent and step, a transition (obs, actions, rewards, terminated, truncated, next_obs, agent). */
	Record,
};

DECLARE_DYNAMIC_MULTICAST_DELEGATE_ThreeParams(FTorchEpisodeSignature, UTorchAgentComponent*, Agent, float, Return, bool, bTerminated);

/**
 * A reinforcement learning environment: the observation and action spec, the agents (Torch Agent Components),
 * the decision steps and the episodes. Train with any Python RL code through unreal_torch.rl.UnrealVectorEnv
 * (a Gymnasium vector env, one sub-env per agent), run the trained policy, or record play for imitation.
 *
 * Steps advance in fixed simulated time (Step Seconds), not frame time: agents integrate with Get Step
 * Seconds, and with a seed every run is reproducible.
 */
UCLASS(BlueprintType)
class PYTORCH_API ATorchEnvironment : public AActor
{
	GENERATED_BODY()
public:
	ATorchEnvironment();

	/** UnrealVectorEnv(EnvName) in Python. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment")
	FString EnvName = TEXT("Env");

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment")
	TArray<FTorchObservationSpec> Observations;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment")
	FTorchActionSpec Actions;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment")
	ETorchEnvironmentMode Mode = ETorchEnvironmentMode::Run;

	/** Decision steps per engine tick (more for kinematic agents that simulate inside Apply Actions; training
	 then runs faster than real time). 0: step only when Step Environment is called. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment", meta = (ClampMin = "0"))
	int32 StepsPerTick = 1;

	/** Simulated seconds per decision step. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment", meta = (ClampMin = "0.001"))
	float StepSeconds = 1.f / 30.f;

	/** Episodes end (truncated) after this many steps; 0: never. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment", meta = (ClampMin = "0"))
	int32 MaxEpisodeSteps = 500;

	/** Seeds the agents' random stream (Random Range) at the start and on resets Python asks for. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment")
	int32 Seed = 0;

	/** Train: the Torch Model whose script defines the training job. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment|Train")
	TObjectPtr<UTorchModel> TrainingModel;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment|Train")
	FString TrainingJob = TEXT("train");

	/** The job's parameters (a JSON object). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment|Train")
	FString TrainingParamsJson;

	/** Run the job in a Python process of its own (recommended: the game's inference doesn't share its GIL). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment|Train")
	bool bTrainInOwnProcess = true;

	/** Start the job when play begins (Train mode). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment|Train")
	bool bStartTrainingOnBeginPlay = true;

	/** The mode when training ends (the job closed its env, finished or failed): Run hands over to the trained
	 Policy Model (Off without one). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment|Train")
	ETorchEnvironmentMode ModeAfterTraining = ETorchEnvironmentMode::Run;

	/** How long a step waits for the trainer's actions before training is abandoned. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment|Train", meta = (ClampMin = "1"))
	float StepTimeoutSeconds = 120.f;

	/** Run: the policy (an exported method taking (agents, observation size) floats, returning actions). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment|Run")
	TObjectPtr<UTorchModel> PolicyModel;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment|Run")
	FString PolicyMethod = TEXT("act");

	/** The policy method's input. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment|Run")
	FString PolicyInput = TEXT("obs");

	/** Record: Saved/Torch/Recordings/<Recording Name>/<timestamp>/. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment|Record")
	FString RecordingName = TEXT("Env");

	/** Steps per recording shard file. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment|Record", meta = (ClampMin = "1"))
	int32 StepsPerShard = 1000;

	UPROPERTY(BlueprintAssignable, Category = "Environment")
	FTorchEpisodeSignature OnEpisodeEnded;

	/** One decision step now (what Tick does Steps Per Tick times). False if nothing happened (Off, no agents,
	 Run waiting for its actions, Train not connected). */
	UFUNCTION(BlueprintCallable, Category = "Environment")
	bool StepEnvironment();

	/** Reset every agent (a new episode for all). */
	UFUNCTION(BlueprintCallable, Category = "Environment")
	void ResetAllAgents();

	/** Reseed the random stream and reset every agent: from here a run replays exactly (Run is turn-based). */
	UFUNCTION(BlueprintCallable, Category = "Environment")
	void ResetWithSeed(int32 NewSeed);

	UFUNCTION(BlueprintCallable, Category = "Environment")
	void SetMode(ETorchEnvironmentMode NewMode);

	/** Start the training job (Train mode; done on BeginPlay by default). */
	UFUNCTION(BlueprintCallable, Category = "Environment")
	UPythonJob* StartTraining();

	UFUNCTION(BlueprintCallable, Category = "Environment")
	void StopTraining();

	UFUNCTION(BlueprintPure, Category = "Environment")
	UPythonJob* GetTrainingJob() const { return TrainingJobHandle; }

	/** Flush the current recording shard to disk (Record mode; also on EndPlay). */
	UFUNCTION(BlueprintCallable, Category = "Environment")
	void FlushRecording();

	UFUNCTION(BlueprintPure, Category = "Environment")
	int32 GetObservationSize() const;

	UFUNCTION(BlueprintPure, Category = "Environment")
	TArray<UTorchAgentComponent*> GetAgents() const { return Agents; }

	/** Decision steps so far, episodes finished, and the mean return of the last 100 episodes. */
	UFUNCTION(BlueprintPure, Category = "Environment")
	void GetStats(int64& OutSteps, int32& OutEpisodes, float& OutMeanReturn) const;

	FRandomStream& GetRandomStream() { return Random; }
	void RegisterAgent(UTorchAgentComponent* Agent);
	void UnregisterAgent(UTorchAgentComponent* Agent);

	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
	virtual void Tick(float DeltaSeconds) override;

private:
	void BuildLayout();
	/** Every agent's observation into Out (rows of obs size); resets finished agents first when bResetDone,
	 recording their last observation, reward and flags. */
	void Collect(TArray<float>& Obs, TArray<float>& FinalObs, TArray<float>& Rewards, TArray<float>& Terminated, TArray<float>& Truncated, bool bResetAll);
	void Apply(const TArray<float>& Flat);
	bool StepTrain();
	bool StepRun();
	bool StepHeuristic(bool bRecord);
	void EndTraining(const TCHAR* Why);
	bool OpenTrainStream();
	void CloseTrainStream();
	void EndEpisodes();
	/** The last step's (obs, actions) completed by this step's rewards and flags (next obs: the final one where
	 an episode ended). */
	void Record(const TArray<float>& Obs, const TArray<float>& FinalObs, const TArray<float>& Rewards, const TArray<float>& Terminated, const TArray<float>& Truncated);

	UPROPERTY()
	TArray<TObjectPtr<UTorchAgentComponent>> AgentRefs;
	TArray<UTorchAgentComponent*> Agents;
	TMap<FString, TPair<int32, int32>> Layout;
	FRandomStream Random;
	TSharedPtr<FPythonStream, ESPMode::ThreadSafe> TrainStream;
	bool bNeedsReset = true;
	bool bRunPending = false;
	TArray<float> RunActions;
	bool bRunActionsReady = false;
	int64 TotalSteps = 0;
	int32 Episodes = 0;
	TArray<float> RecentReturns;
	int32 RunRequestGeneration = 0;

	UPROPERTY()
	TObjectPtr<UPythonJob> TrainingJobHandle;

	struct FRecordingBuffer
	{
		FString Directory;
		int32 Shard = 0;
		int32 Steps = 0;
		TArray<float> Obs, Actions, Rewards, Terminated, Truncated, NextObs;
		TArray<int32> AgentIds;
		/** The step waiting for its rewards. */
		TArray<float> PendingObs, PendingActions;
	};
	FRecordingBuffer Recording;
};
