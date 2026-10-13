#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintAsyncActionBase.h"
#include "TorchTypes.h"
#include "TorchAsyncActions.generated.h"

class UTorchModel;

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FTorchResultSignature, const FTorchResult&, Result);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FTorchLoadSignature, const FString&, Error);

/** Blueprint: Run Model. */
UCLASS()
class PYTORCH_API UTorchRunModelAsyncAction : public UBlueprintAsyncActionBase
{
	GENERATED_BODY()
public:
	UPROPERTY(BlueprintAssignable)
	FTorchResultSignature OnSuccess;

	/** Result's Error and Traceback say why. */
	UPROPERTY(BlueprintAssignable)
	FTorchResultSignature OnFailure;

	/** Run an exported method of a Torch Model. Inputs: tensors by input name (one request, no batch dimension);
	 JsonInputs: a JSON object with the other inputs ({"top_k": 3}). Requests in the same tick run together. */
	UFUNCTION(BlueprintCallable, Category = "Torch", meta = (BlueprintInternalUseOnly = "true", AutoCreateRefTerm = "Inputs,JsonInputs", DisplayName = "Run Model"))
	static UTorchRunModelAsyncAction* RunModel(UTorchModel* Model, const FString& Method, const TMap<FString, FPyTensor>& Inputs, const FString& JsonInputs);

	/** Run Model with one float input (Values in row-major order, Shape without the batch dimension; empty Shape:
	 [Num]). Read the output with Get Output Floats. */
	UFUNCTION(BlueprintCallable, Category = "Torch", meta = (BlueprintInternalUseOnly = "true", AutoCreateRefTerm = "Shape", DisplayName = "Run Model (Floats)"))
	static UTorchRunModelAsyncAction* RunModelWithFloats(UTorchModel* Model, const FString& Method, const FString& InputName, const TArray<float>& Values, const TArray<int64>& Shape);

	virtual void Activate() override;

private:
	TWeakObjectPtr<UTorchModel> Model;
	FString Method;
	TMap<FString, FPyTensor> Inputs;
	FString JsonInputs;
};

/** Blueprint: Load Model (for loading screens, or Load = Manual). */
UCLASS()
class PYTORCH_API UTorchLoadModelAsyncAction : public UBlueprintAsyncActionBase
{
	GENERATED_BODY()
public:
	UPROPERTY(BlueprintAssignable)
	FTorchLoadSignature OnLoaded;

	UPROPERTY(BlueprintAssignable)
	FTorchLoadSignature OnFailed;

	/** Load a Torch Model now (installing the Python environment first if needed). */
	UFUNCTION(BlueprintCallable, Category = "Torch", meta = (BlueprintInternalUseOnly = "true", DisplayName = "Load Model"))
	static UTorchLoadModelAsyncAction* LoadTorchModel(UTorchModel* Model);

	virtual void Activate() override;

private:
	TWeakObjectPtr<UTorchModel> Model;
};

class UPythonJob;
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FTorchJobActionSignature, UPythonJob*, Job);

/** Blueprint: Start Job, with a pin for each stage of the job. */
UCLASS()
class PYTORCH_API UTorchStartJobAsyncAction : public UBlueprintAsyncActionBase
{
	GENERATED_BODY()
public:
	/** Right away, with the job (keep it to Stop it). */
	UPROPERTY(BlueprintAssignable)
	FTorchJobActionSignature OnStarted;

	/** Job's Progress / Progress Message changed. */
	UPROPERTY(BlueprintAssignable)
	FTorchJobActionSignature OnProgress;

	/** job.metrics(): Job's Last Metrics Json / Get Metric. */
	UPROPERTY(BlueprintAssignable)
	FTorchJobActionSignature OnMetrics;

	/** The job returned (State Completed) or ended after Stop (State Stopped): Job's Result Json. */
	UPROPERTY(BlueprintAssignable)
	FTorchJobActionSignature OnCompleted;

	/** Job's Error / Traceback. */
	UPROPERTY(BlueprintAssignable)
	FTorchJobActionSignature OnFailed;

	/** Start a @ut.job of a Torch Model's script. ParamsJson: a JSON object of its parameters ({"epochs": 2});
	 Resume continues its latest run; Own Process runs it in a Python process of its own. */
	UFUNCTION(BlueprintCallable, Category = "Torch", meta = (BlueprintInternalUseOnly = "true", AutoCreateRefTerm = "ParamsJson", DisplayName = "Start Job"))
	static UTorchStartJobAsyncAction* StartTorchJob(UTorchModel* Model, const FString& JobName, const FString& ParamsJson, bool bResume, bool bOwnProcess);

	virtual void Activate() override;

private:
	UFUNCTION()
	void HandleProgress(UPythonJob* InJob, float Progress, const FString& Message);
	UFUNCTION()
	void HandleMetrics(UPythonJob* InJob, int32 Step, const FString& MetricsJson);
	UFUNCTION()
	void HandleFinished(UPythonJob* InJob, const FString& ResultJson);
	UFUNCTION()
	void HandleFailed(UPythonJob* InJob, const FString& Error, const FString& Traceback);
	void Done();

	TWeakObjectPtr<UTorchModel> Model;
	FString JobName;
	FString ParamsJson;
	bool bResume = false;
	bool bOwnProcess = false;
	UPROPERTY()
	TObjectPtr<UPythonJob> Job;
};
