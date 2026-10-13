#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "TorchTypes.h"
#include "TorchComponent.generated.h"

class UPythonJob;
class UTextureRenderTarget2D;
class UTorchModel;

DECLARE_DYNAMIC_MULTICAST_DELEGATE_ThreeParams(FTorchComponentResultSignature, int32, RequestId, const FString&, Method, const FTorchResult&, Result);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FTorchComponentLoadedSignature, bool, bSuccess, const FString&, Error);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_ThreeParams(FTorchComponentJobProgressSignature, UPythonJob*, Job, float, Progress, const FString&, Message);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FTorchComponentJobMetricsSignature, UPythonJob*, Job, const FString&, MetricsJson);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FTorchComponentJobFinishedSignature, UPythonJob*, Job, const FString&, ResultText);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FTorchComponentJobFailedSignature, UPythonJob*, Job, const FString&, Error);

/**
 * A Torch Model on an actor, with events instead of latent nodes: Run sends a request and On Result answers it
 * (with the request's id and method); Start Job reports through On Job Progress / Metrics / Finished / Failed.
 * The model itself is shared by everything that uses it (UTorchSubsystem); this component is a handle to it.
 */
UCLASS(ClassGroup = "Torch", meta = (BlueprintSpawnableComponent))
class PYTORCH_API UTorchComponent : public UActorComponent
{
	GENERATED_BODY()
public:
	UTorchComponent();

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Torch")
	TObjectPtr<UTorchModel> Model;

	/** Load the model on BeginPlay (On Model Loaded tells when it's ready). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Torch")
	bool bLoadOnBeginPlay = true;

	/** Stop the jobs this component started when it ends. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Torch")
	bool bStopJobsOnEndPlay = true;

	UPROPERTY(BlueprintAssignable, Category = "Torch")
	FTorchComponentResultSignature OnResult;

	UPROPERTY(BlueprintAssignable, Category = "Torch")
	FTorchComponentLoadedSignature OnModelLoaded;

	UPROPERTY(BlueprintAssignable, Category = "Torch|Jobs")
	FTorchComponentJobProgressSignature OnJobProgress;

	UPROPERTY(BlueprintAssignable, Category = "Torch|Jobs")
	FTorchComponentJobMetricsSignature OnJobMetrics;

	/** Completed or stopped (the job's State says which). ResultText: the result's "text" if it has one, else the
	 result as JSON. */
	UPROPERTY(BlueprintAssignable, Category = "Torch|Jobs")
	FTorchComponentJobFinishedSignature OnJobFinished;

	UPROPERTY(BlueprintAssignable, Category = "Torch|Jobs")
	FTorchComponentJobFailedSignature OnJobFailed;

	/** Run an exported method; On Result gets the answer with the returned id. */
	UFUNCTION(BlueprintCallable, Category = "Torch", meta = (AutoCreateRefTerm = "Inputs,JsonInputs"))
	int32 Run(const FString& Method, const TMap<FString, FPyTensor>& Inputs, const FString& JsonInputs);

	/** Run with one tensor input (InputName empty: no tensor input, just JsonInputs). */
	UFUNCTION(BlueprintCallable, Category = "Torch", meta = (AutoCreateRefTerm = "InputName,Tensor,JsonInputs"))
	int32 RunWithTensor(const FString& Method, const FString& InputName, const FPyTensor& Tensor, const FString& JsonInputs);

	/** Run with a render target's pixels as an image input (read now: blocking, small targets; use a stream for
	 every frame of a large one). */
	UFUNCTION(BlueprintCallable, Category = "Torch", meta = (AutoCreateRefTerm = "JsonInputs"))
	int32 RunWithRenderTarget(const FString& Method, const FString& InputName, UTextureRenderTarget2D* RenderTarget, const FString& JsonInputs);

	/** Start a @ut.job of the model's script (see UTorchSubsystem::StartJob). */
	UFUNCTION(BlueprintCallable, Category = "Torch|Jobs", meta = (AutoCreateRefTerm = "ParamsJson"))
	UPythonJob* StartJob(const FString& JobName, const FString& ParamsJson, bool bResume, bool bOwnProcess = false);

	/** Stop the jobs this component started. */
	UFUNCTION(BlueprintCallable, Category = "Torch|Jobs")
	void StopJobs();

	/** The last job this component started (running or not). */
	UFUNCTION(BlueprintPure, Category = "Torch|Jobs")
	UPythonJob* GetLastJob() const { return LastJob; }

	UFUNCTION(BlueprintPure, Category = "Torch|Jobs")
	bool IsJobRunning() const;

	UFUNCTION(BlueprintPure, Category = "Torch")
	bool IsModelReady() const;

	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

private:
	UFUNCTION()
	void HandleJobProgress(UPythonJob* Job, float Progress, const FString& Message);
	UFUNCTION()
	void HandleJobMetrics(UPythonJob* Job, int32 Step, const FString& MetricsJson);
	UFUNCTION()
	void HandleJobFinished(UPythonJob* Job, const FString& ResultJson);
	UFUNCTION()
	void HandleJobFailed(UPythonJob* Job, const FString& Error, const FString& Traceback);

	int32 NextRequestId = 0;

	UPROPERTY()
	TArray<TObjectPtr<UPythonJob>> Jobs;

	UPROPERTY()
	TObjectPtr<UPythonJob> LastJob;
};
