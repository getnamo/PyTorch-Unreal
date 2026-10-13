#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "TorchAgentComponent.h"
#include "TorchEnvironment.h"
#include "TorchReachExample.generated.h"

class UPythonJob;
class UStaticMeshComponent;
class UTextRenderComponent;

/**
 * The reach task's agent: a point that moves on a square arena (velocity control, 2 actions in [-1, 1]) and is
 * rewarded for getting to its target. Observations: the offset to the target and its velocity (4 values).
 * Rewards: progress toward the target each step, -0.01 per step, +1 on arrival (the episode ends, terminated).
 * Kinematic: it moves Step Seconds per decision step, so training can run many steps per frame.
 */
UCLASS(ClassGroup = "Torch")
class PYTORCHEXAMPLES_API UTorchReachAgentComponent : public UTorchAgentComponent
{
	GENERATED_BODY()
public:
	/** Half the arena's width (units). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Reach")
	float ArenaHalfSize = 300.f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Reach")
	float MaxSpeed = 400.f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Reach")
	float ReachRadius = 30.f;

	UPROPERTY(BlueprintReadOnly, Category = "Reach")
	FVector2D Position = FVector2D::ZeroVector;

	UPROPERTY(BlueprintReadOnly, Category = "Reach")
	FVector2D Velocity = FVector2D::ZeroVector;

	UPROPERTY(BlueprintReadOnly, Category = "Reach")
	FVector2D Target = FVector2D::ZeroVector;

	virtual void CollectObservations() override;
	virtual void ApplyActions(const TArray<float>& InActions) override;
	virtual void ResetAgent() override;
	/** A scripted expert: straight to the target (for recording demonstrations). */
	virtual void Heuristic() override;

private:
	void UpdateVisuals();
};

/** An agent's actor: a sphere for the agent (orange), a cube for its target (green), on its own arena. */
UCLASS()
class PYTORCHEXAMPLES_API ATorchReachAgent : public AActor
{
	GENERATED_BODY()
public:
	ATorchReachAgent();
	virtual void PostInitializeComponents() override;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Reach")
	TObjectPtr<UStaticMeshComponent> Body;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Reach")
	TObjectPtr<UStaticMeshComponent> TargetMarker;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Reach")
	TObjectPtr<UStaticMeshComponent> Floor;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Reach")
	TObjectPtr<UTorchReachAgentComponent> Agent;
};

/**
 * The reach task's environment ("Reach"): spawns Num Agents agents on a grid of arenas, with the reach spec
 * (4 observations, 2 continuous actions). Train it with examples/reach.py's train_reach (PPO), run the trained
 * reach_policy, or record the scripted expert for behavior cloning (train_reach_bc).
 *
 * The example map's controls (Example Controls): [T] train, [R] run the policy, [H] the scripted expert, [C]
 * record it, [B] behavior cloning from the latest recording, [1] / [2] replay seed 1 / 2. A status text shows the
 * mode, the steps, the mean return and the training job's progress.
 */
UCLASS()
class PYTORCHEXAMPLES_API ATorchReachEnvironment : public ATorchEnvironment
{
	GENERATED_BODY()
public:
	ATorchReachEnvironment();

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Reach", meta = (ClampMin = "1"))
	int32 NumAgents = 16;

	/** Distance between arenas. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Reach")
	float Spacing = 700.f;

	/** Keyboard controls and the status text (the example map). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Reach")
	bool bExampleControls = true;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Reach")
	TObjectPtr<UTextRenderComponent> StatusText;

	/** Spawn the agents (done on BeginPlay; call it yourself in a world that doesn't begin play). */
	UFUNCTION(BlueprintCallable, Category = "Reach")
	void SpawnAgents();

	/** Behavior cloning ([B]): train_reach_bc on the latest recording, in its own process. */
	UFUNCTION(BlueprintCallable, Category = "Reach")
	UPythonJob* StartBehaviorCloning();

	/** The status line (what the status text shows). */
	UFUNCTION(BlueprintPure, Category = "Reach")
	FString GetStatus() const;

	virtual void BeginPlay() override;
	virtual void Tick(float DeltaSeconds) override;
	virtual void EnableInput(APlayerController* PlayerController) override;

private:
	void OnKey(FKey Key);

	bool bSpawned = false;
	bool bKeysBound = false;
	/** The status shows the cloning job (started last) rather than the training job. */
	bool bShowCloning = false;
	float StatusRefresh = 0.f;

	UPROPERTY()
	TObjectPtr<UPythonJob> CloningJob;
};
