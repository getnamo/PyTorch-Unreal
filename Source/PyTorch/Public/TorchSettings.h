#pragma once

#include "CoreMinimal.h"
#include "Engine/DeveloperSettings.h"
#include "TorchSettings.generated.h"

/** Which PyTorch build the plugin installs into the project's Python environment. */
UENUM(BlueprintType)
enum class ETorchProfile : uint8
{
	/** The CUDA build matching the installed NVIDIA driver (uv --torch-backend auto); CPU when there's no NVIDIA
	 GPU. On Windows also triton-windows, so torch.compile works on the GPU. */
	CUDAAuto UMETA(DisplayName = "CUDA (Auto)"),
	CUDA130 UMETA(DisplayName = "CUDA 13.0"),
	CUDA132 UMETA(DisplayName = "CUDA 13.2"),
	/** Small; no GPU. */
	CPU,
	/** AMD GPUs on Linux (best effort): uv picks the ROCm build. */
	ROCm UMETA(DisplayName = "ROCm (Linux)"),
	/** Intel GPUs (best effort). */
	XPU,
	/** Nothing: list your own packages in Project Settings > Plugins > Python CLI > Requirements. */
	None,
};

/** What packaged builds that ship Python (Python CLI > Packaged Env Mode: Bundle or Install On First Run) get.
 Builds without Python (Packaged Env Mode: None) run the Torch Models' NNE exports. */
UENUM(BlueprintType)
enum class ETorchPackagedProfile : uint8
{
	/** CPU torch (+ numpy, safetensors): hundreds of MB instead of gigabytes. GPU inference in packaged games
	 goes through exported ONNX models on NNE. */
	CPU,
	/** The editor's environment as it is (CUDA builds are several GB). */
	SameAsEditor,
};

/** Project Settings > Plugins > PyTorch. */
UCLASS(config = Game, defaultconfig, meta = (DisplayName = "PyTorch"))
class PYTORCH_API UTorchSettings : public UDeveloperSettings
{
	GENERATED_BODY()
public:
	UTorchSettings();

	/** The PyTorch build installed into the project's Python environment (installs are additive: switching
	 keeps what's installed until the venv is deleted). */
	UPROPERTY(config, EditAnywhere, Category = "Environment")
	ETorchProfile Profile = ETorchProfile::CUDAAuto;

	UPROPERTY(config, EditAnywhere, Category = "Environment|Extras")
	bool bTorchvision = true;

	UPROPERTY(config, EditAnywhere, Category = "Environment|Extras")
	bool bTorchaudio = false;

	/** onnx + onnxscript: exporting models to ONNX for NNE (shipping without Python). */
	UPROPERTY(config, EditAnywhere, Category = "Environment|Extras")
	bool bOnnxExport = true;

	/** gymnasium + pettingzoo: reinforcement learning environments. */
	UPROPERTY(config, EditAnywhere, Category = "Environment|Extras")
	bool bGymnasium = false;

	UPROPERTY(config, EditAnywhere, Category = "Environment|Extras")
	bool bTensorBoard = false;

	/** huggingface_hub: hf:// weights. */
	UPROPERTY(config, EditAnywhere, Category = "Environment|Extras")
	bool bHuggingFaceHub = false;

	UPROPERTY(config, EditAnywhere, Category = "Packaging")
	ETorchPackagedProfile PackagedProfile = ETorchPackagedProfile::CPU;

	/** NNE runtimes for a Torch Model's GPU runs, tried in order (set per platform in Config/<Platform>/
	 <Platform>Game.ini). Default: ORT DirectML (Win64). */
	UPROPERTY(config, EditAnywhere, Category = "Shipping")
	TArray<FString> NNEGPURuntimes;

	/** NNE runtimes for CPU runs, tried in order. Default: ORT CPU. */
	UPROPERTY(config, EditAnywhere, Category = "Shipping")
	TArray<FString> NNECPURuntimes;

	/** torch.compile's kernel caches (Inductor, Triton), kept between editor sessions. Relative to the project. */
	UPROPERTY(config, EditAnywhere, Category = "Performance")
	FString CompileCacheDirectory = TEXT("Intermediate/Torch/CompileCache");

	/** Register the profile's requirements, the torch backend and the environment variables with PythonCLI. */
	void ApplyProfile() const;

	/** The uv --torch-backend of a profile (empty for None). */
	static FString TorchBackend(ETorchProfile InProfile);

	virtual FName GetCategoryName() const override { return TEXT("Plugins"); }
#if WITH_EDITOR
	virtual void PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) override;
#endif
};
