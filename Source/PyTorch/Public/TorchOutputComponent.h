#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "TorchTypes.h"
#include "TorchOutputComponent.generated.h"

class UPythonStreamComponent;
class UTextureRenderTarget2D;
class UTorchModel;

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FTorchOutputSignature, const FTorchResult&, Result);

/**
 * Runs a model's method on every frame of a stream, with no script and no Blueprint graph needed: point it at a
 * Python stream component on the same actor (render target, audio, observations) and a Torch Model method. The
 * outputs arrive in On Output, and image outputs can go straight into render targets (Render Target Sinks), for
 * image-to-image effects. Camera (Latest) streams keep one frame in flight: results never lag behind.
 */
UCLASS(ClassGroup = "Torch", meta = (BlueprintSpawnableComponent))
class PYTORCH_API UTorchOutputComponent : public UActorComponent
{
	GENERATED_BODY()
public:
	UTorchOutputComponent();

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Torch")
	TObjectPtr<UTorchModel> Model;

	/** The exported method each frame runs. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Torch")
	FString Method;

	/** The method's input the frame goes to (empty: its first input). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Torch")
	FString InputName;

	/** The method's other inputs, as a JSON object ({"threshold": 0.5}). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Torch")
	FString JsonInputs;

	/** The stream component on this actor whose frames go to the model (its component name). Empty: the first
	 Python stream component on the actor. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Torch")
	FName StreamComponent;

	/** Image outputs written into render targets after each frame: output name -> render target. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Torch")
	TMap<FString, TObjectPtr<UTextureRenderTarget2D>> RenderTargetSinks;

	/** Image outputs are (C, H, W), torch's layout (off: (H, W, C)). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Torch", AdvancedDisplay)
	bool bChannelsFirst = true;

	/** Bind the stream when play begins (else call Bind). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Torch")
	bool bBindOnBeginPlay = true;

	/** Every result (On Output also fires for failures: Result.bSuccess is false). */
	UPROPERTY(BlueprintAssignable, Category = "Torch")
	FTorchOutputSignature OnOutput;

	UPROPERTY(BlueprintReadOnly, Category = "Torch")
	FTorchResult LastResult;

	/** Results so far. */
	UPROPERTY(BlueprintReadOnly, Category = "Torch")
	int32 ResultCount = 0;

	/** Send the stream's frames to the model (reopens the stream). */
	UFUNCTION(BlueprintCallable, Category = "Torch")
	bool Bind();

	UFUNCTION(BlueprintCallable, Category = "Torch")
	void Unbind();

	/** Results per second over the last second or so. */
	UFUNCTION(BlueprintPure, Category = "Torch")
	float GetResultsPerSecond() const;

	/** The stream component the frames come from. */
	UPythonStreamComponent* ResolveStream() const;

	/** UTorchSubsystem: a result for this binding (game thread). */
	void HandleResult(const FTorchResult& Result);

	virtual void InitializeComponent() override;
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

private:
	/** Point the stream at the model's service (before it opens). */
	bool SetTarget();

	int32 BindingId = 0;
	TArray<double> RecentResults;
};
