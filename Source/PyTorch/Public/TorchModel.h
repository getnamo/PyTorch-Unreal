#pragma once

#include "CoreMinimal.h"
#include "Engine/DataAsset.h"
#include "PythonService.h"
#include "TorchTypes.h"
#include "TorchModel.generated.h"

class UNNEModelData;

/** What runs a Torch Model's methods. */
UENUM(BlueprintType)
enum class ETorchBackend : uint8
{
	/** Python where it's available (the editor; packaged builds that ship Python), else the NNE export. */
	Auto,
	Python,
	/** The NNE export (Export To NNE), in the editor too. */
	NNE,
};

/** Which NNE runtime runs the export (the runtimes per platform: Project Settings > Plugins > PyTorch). */
UENUM(BlueprintType)
enum class ETorchNNERuntime : uint8
{
	/** Time the GPU and CPU runtimes on the first run and keep the faster: the GPU (DirectML) wins on image
	 models, the CPU on small MLPs. */
	Auto,
	GPU,
	CPU,
};

/** One exported method: ONNX models (NNE model data assets) and how to feed them. */
USTRUCT(BlueprintType)
struct PYTORCH_API FTorchNNEMethod
{
	GENERATED_BODY()

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Torch")
	FString Method;

	/** Any batch size (ORT CPU; GPU runtimes too, without a static model). */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Torch")
	TObjectPtr<UNNEModelData> Model;

	/** A fixed batch of Static Batch rows: GPU runtimes run requests padded (or in chunks) to it, since
	 DirectML is fast only with static shapes. */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Torch")
	TObjectPtr<UNNEModelData> StaticModel;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Torch")
	int32 StaticBatch = 0;

	/** The export's description (JSON): inputs (per-row shapes; images: how Unreal converts them), outputs. */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Torch")
	FString SpecJson;

	/** Largest difference from torch on the export's reference rows (ORT CPU; -1: not checked). */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Torch")
	float ParityMaxDiff = -1.f;
};

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

	/** Python, NNE (the export below), or Auto: Python where it's available, else NNE (packaged builds without
	 Python). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Shipping")
	ETorchBackend Backend = ETorchBackend::Auto;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Shipping")
	ETorchNNERuntime NNERuntime = ETorchNNERuntime::Auto;

	/** Export To NNE: the methods to export (empty: every batch=True method). */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Shipping")
	TArray<FString> ExportMethods;

	/** Export To NNE also writes a static-batch model of this many rows, for GPU runtimes (0: none). Worth it for
	 image models called with more than one row at a time. */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Shipping", meta = (ClampMin = "0"))
	int32 ExportStaticBatch = 0;

	/** The NNE export: ONNX models of the exported methods, next to this asset (Export To NNE). */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Shipping")
	TArray<FTorchNNEMethod> NNEMethods;

	const FTorchNNEMethod* FindNNEMethod(const FString& Name) const;

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

	/** Export the methods to ONNX (ut.export.onnx) and import them as NNE model data assets next to this one
	 (<Asset>_<method>), checking them against torch. Packaged builds then run them without Python. */
	UFUNCTION(CallInEditor, Category = "Shipping")
	void ExportToNNE();
#endif
};
