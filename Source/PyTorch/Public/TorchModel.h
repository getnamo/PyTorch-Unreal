#pragma once

#include "CoreMinimal.h"
#include "Engine/DataAsset.h"
#include "PythonService.h"
#include "TorchTypes.h"
#include "TorchModel.generated.h"

/**
 * A PyTorch model: the @ut.model class in a Python script (Script + Model Name) and how it runs. Use it with
 * Run Model; every caller shares one instance, and requests made in the same tick run as one batch.
 */
UCLASS(BlueprintType)
class PYTORCH_API UTorchModel : public UDataAsset
{
	GENERATED_BODY()
public:
	/** The model's script, under Content/Scripts/ (or a plugin's scripts, e.g. "examples/digits.py"). */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Model")
	FString Script;

	/** The @ut.model name. Empty: this asset's name. */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Model")
	FString ModelName;

	/** Shared: the engine's warm Python process (with the other models). Dedicated: a process of its own. */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Model")
	EPythonServiceProcess Process = EPythonServiceProcess::Shared;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Model")
	ETorchDevice Device = ETorchDevice::Default;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Model")
	ETorchPrecision Precision = ETorchPrecision::Default;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Model")
	ETorchCompile Compile = ETorchCompile::Default;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, AssetRegistrySearchable, Category = "Model")
	ETorchLoad Load = ETorchLoad::OnFirstUse;

	/** Run each exported method once after loading (CUDA kernels, cuDNN tuning, compile) so the first real call
	 is fast. */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Model")
	bool bWarmup = true;

	/** Send the requests of a tick together (one call, stacked on dim 0). Off: one call per request. */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Model", AdvancedDisplay)
	bool bStackRequests = true;

	/** Most requests stacked into one call. */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Model", AdvancedDisplay, meta = (ClampMin = "1"))
	int32 MaxStackedRequests = 1024;

	/** The script's exported methods (refreshed from Python: Refresh Signature). */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Signature")
	TArray<FTorchMethodSpec> Methods;

	/** The @ut.job functions in the model's script. */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Signature")
	TArray<FTorchJobSpec> Jobs;

	/** The @ut.model name to use. */
	UFUNCTION(BlueprintPure, Category = "Torch")
	FString GetModelName() const;

	/** The method's spec, or null. */
	const FTorchMethodSpec* FindMethod(const FString& Name) const;

	/** The options _torch_load gets (device, precision, compile, warmup). */
	TSharedRef<class FJsonObject> MakeLoadOptions() const;

	/** Fill Methods and Jobs from the script's signature JSON (_torch_list / _torch_signature). */
	void ApplySignature(const TSharedPtr<class FJsonObject>& ModelSignature, const TArray<TSharedPtr<class FJsonValue>>* JobList);

#if WITH_EDITOR
	/** Ask Python for the script's methods and jobs (starts the model's script). */
	UFUNCTION(CallInEditor, Category = "Signature")
	void RefreshSignature();
#endif
};
