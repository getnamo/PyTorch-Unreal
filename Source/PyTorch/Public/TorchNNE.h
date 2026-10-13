#pragma once

#include "CoreMinimal.h"
#include "PythonTypes.h"

class FJsonObject;
class UTorchModel;

/**
 * Runs Torch Models' NNE exports (ONNX models on NNE's ORT DirectML / CPU runtimes) behind the same requests as
 * Python: UTorchSubsystem routes a model here when its backend is NNE, or Auto without Python. Inputs arrive as
 * Unreal sends them (images raw: (N, H, W, C) uint8 or float) and are converted the way the method's ut.Image
 * would; runs happen on a worker thread, one at a time per method.
 */
class PYTORCH_API FTorchNNE
{
public:
	struct FResult
	{
		bool bSuccess = false;
		FString Error;
		/** Output names and tensors, with the batch dimension. */
		TArray<FString> Names;
		TArray<FPyTensor> Outputs;
		float Ms = 0.f;
		/** The NNE runtime that ran it. */
		FString Runtime;
	};

	static FTorchNNE& Get();

	/** Run stacked rows (each input with its batch dimension of Rows) on a worker thread; OnDone on the game thread.
	 Layouts: channel orders of raw image inputs ("BGRA", ...). */
	void Run(UTorchModel* Model, const FString& Method, TMap<FString, FPyTensor>&& Inputs, int64 Rows, const TMap<FString, FString>& Layouts,
		TFunction<void(FResult&&)> OnDone);

	/** Run now, on this thread, inputs the model takes as they are (in the export's input order). ForceRuntime:
	 a runtime name, or empty for the model's choice. */
	FResult RunNow(UTorchModel* Model, const FString& Method, const TArray<FPyTensor>& ModelInputs, const FString& ForceRuntime = FString());

	/** Create the method's runtime instances now (otherwise on first use). */
	bool Prepare(UTorchModel* Model, const FString& Method, FString& OutError);

	/** Forget a model's instances (after a new export). */
	void Reset(const UTorchModel* Model);

	/** The runtime the method runs on (empty before its first run). */
	FString GetRuntime(const UTorchModel* Model, const FString& Method) const;

	/** One raw input (Rows rows) converted to what the model takes, per its spec from the export. */
	static bool ConvertInput(const FJsonObject& Spec, const FPyTensor& In, int64 Rows, const FString& Layout, FPyTensor& Out, FString& OutError);

	/** A safetensors file's tensors. */
	static bool ReadSafetensors(const FString& Path, TMap<FString, FPyTensor>& Out, FString& OutError);

	struct FInstance;
	struct FState;

private:
	TSharedPtr<FState> FindOrCreate(UTorchModel* Model, const FString& Method, FString& OutError);
	static FResult Execute(FState& State, const TArray<FPyTensor>& ModelInputs, const FString& ForceRuntime);

	TMap<FString, TSharedPtr<FState>> States;
};
