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
