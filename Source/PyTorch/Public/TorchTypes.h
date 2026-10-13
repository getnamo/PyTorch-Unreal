#pragma once

#include "CoreMinimal.h"
#include "PythonTypes.h"
#include "TorchTypes.generated.h"

/** Where a model runs (overrides the script's @ut.model(device=...)). */
UENUM(BlueprintType)
enum class ETorchDevice : uint8
{
	/** What the script asks for (default: Auto). */
	Default,
	/** CUDA if available, then XPU, then the CPU. */
	Auto,
	CUDA,
	CPU,
};

/** Autocast around the model's exported calls. */
UENUM(BlueprintType)
enum class ETorchPrecision : uint8
{
	Default,
	FP32,
	BF16,
	FP16,
};

/** torch.compile for the model's networks (CUDA on Windows uses triton-windows). */
UENUM(BlueprintType)
enum class ETorchCompile : uint8
{
	/** What the script asks for. */
	Default,
	Off,
	/** mode="default": larger models, longer first call (a cold compile takes seconds per input shape). */
	On,
	/** mode="reduce-overhead" (CUDA graphs): small per-step calls such as policies. */
	ReduceOverhead,
};

UENUM(BlueprintType)
enum class ETorchLoad : uint8
{
	/** On the first Run Model (or Load Model). */
	OnFirstUse,
	/** When the engine starts (assets found by the asset registry). */
	OnStartup,
	/** Only by Load Model. */
	Manual,
};

UENUM(BlueprintType)
enum class ETorchModelState : uint8
{
	NotLoaded,
	Loading,
	Ready,
	Failed,
};

/** One input of an exported method (from the script's annotations). */
USTRUCT(BlueprintType)
struct PYTORCH_API FTorchInputSpec
{
	GENERATED_BODY()

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Torch")
	FString Name;

	/** tensor, image, audio or json. */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Torch")
	FString Kind;

	/** Tensors: dtype; JSON: the Python type. */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Torch")
	FString Type;

	/** Per request, without the batch dimension; -1 is any size. Images: (H, W). */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Torch")
	TArray<int64> Shape;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Torch")
	bool bRequired = true;

	/** Everything else the script declared (normalization, sample rate, ...), as JSON. */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Torch")
	FString Details;
};

/** An exported method of a model. */
USTRUCT(BlueprintType)
struct PYTORCH_API FTorchMethodSpec
{
	GENERATED_BODY()

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Torch")
	FString Name;

	/** Concurrent requests run as one call (stacked on dim 0). */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Torch")
	bool bBatched = false;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Torch")
	TArray<FTorchInputSpec> Inputs;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Torch")
	FString Doc;
};

/** A @ut.job in the model's script. */
USTRUCT(BlueprintType)
struct PYTORCH_API FTorchJobSpec
{
	GENERATED_BODY()

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Torch")
	FString Name;

	/** [{name, type, default, required}, ...] */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Torch")
	FString ParamsJson;

	/** "shared" or "dedicated" (@ut.job(process=...)). */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Torch")
	FString Process;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Torch")
	FString Doc;
};

/** A Run Model request's outputs. */
USTRUCT(BlueprintType)
struct PYTORCH_API FTorchResult
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadOnly, Category = "Torch")
	bool bSuccess = false;

	/** Tensor outputs by name ("output" when the method returns one tensor). This request's rows only, without a
	 batch dimension unless the request was batched. */
	UPROPERTY(BlueprintReadOnly, Category = "Torch")
	TMap<FString, FPyTensor> Outputs;

	/** The other outputs (numbers, strings, lists) as a JSON object. */
	UPROPERTY(BlueprintReadOnly, Category = "Torch")
	FString OutputsJson;

	UPROPERTY(BlueprintReadOnly, Category = "Torch")
	FString Error;

	UPROPERTY(BlueprintReadOnly, Category = "Torch")
	FString Traceback;

	/** From the request to the result on the game thread. */
	UPROPERTY(BlueprintReadOnly, Category = "Torch")
	float LatencyMs = 0.f;

	/** Rows in the forward pass that answered this request (requests are batched together). */
	UPROPERTY(BlueprintReadOnly, Category = "Torch")
	int32 BatchSize = 0;

	/** The forward pass's time in Python. */
	UPROPERTY(BlueprintReadOnly, Category = "Torch")
	float ModelMs = 0.f;

	/** "python" or the NNE runtime that ran it. */
	UPROPERTY(BlueprintReadOnly, Category = "Torch")
	FString Backend;

	/** Parsed OutputsJson (C++). */
	TSharedPtr<class FJsonObject> Json;
};

/** Per-request options (C++). */
struct FTorchRunOptions
{
	/** The tensors already have a leading batch dimension (N rows); the outputs keep it. */
	bool bBatched = false;
	/** Channel order of image inputs that aren't RGB(A), e.g. {"image": "BGRA"} for render target pixels. */
	TMap<FString, FString> Layouts;
	/** Send at once instead of with the tick's other requests. */
	bool bImmediate = false;
};
