#pragma once

#include "CoreMinimal.h"
#include "Subsystems/EngineSubsystem.h"
#include "PythonTypes.h"
#include "TorchTypes.h"
#include "TorchSubsystem.generated.h"

class UPythonJob;
class UPythonService;
class UTorchModel;
class UTorchOutputComponent;

DECLARE_DYNAMIC_MULTICAST_DELEGATE_ThreeParams(FTorchModelLoadedSignature, UTorchModel*, Model, bool, bSuccess, const FString&, Error);

/**
 * Runs Torch Model assets. Each model script runs as a PythonCLI service (shared process by default) and its
 * models stay loaded for the engine's lifetime, across PIE sessions. Requests made in the same tick for the
 * same model and method are stacked on dim 0 and sent as one call at the end of the frame; Python may merge
 * calls further (micro-batching) and the results are split back per request.
 */
UCLASS()
class PYTORCH_API UTorchSubsystem : public UEngineSubsystem
{
	GENERATED_BODY()
public:
	static UTorchSubsystem* Get();

	virtual void Initialize(FSubsystemCollectionBase& Collection) override;
	virtual void Deinitialize() override;

	/** Run a model's exported method. Inputs: tensors by name (one request: no batch dimension unless
	 Options.bBatched); JsonInputs: a JSON object of the other inputs. OnResult runs on the game thread.
	 Loads the model first if needed. Blueprint: Run Model. */
	void Run(UTorchModel* Model, const FString& Method, const TMap<FString, FPyTensor>& Inputs, const FString& JsonInputs,
		TFunction<void(const FTorchResult&)> OnResult, const FTorchRunOptions& Options = FTorchRunOptions());

	/** Send the requests collected this frame now (normally at the end of the frame). */
	UFUNCTION(BlueprintCallable, Category = "Torch")
	void Flush();

	/** Load the model with the asset's options (device, precision, compile, warmup); OnDone on the game thread. */
	void LoadModel(UTorchModel* Model, TFunction<void(bool bOk, const FString& Error)> OnDone = nullptr);

	/** Free the model's memory (GPU included). The next request loads it again. */
	UFUNCTION(BlueprintCallable, Category = "Torch")
	void UnloadModel(UTorchModel* Model);

	/** Re-read the model's script; its models keep their weights where the shapes still match. */
	UFUNCTION(BlueprintCallable, Category = "Torch")
	void ReloadScript(UTorchModel* Model);

	UFUNCTION(BlueprintPure, Category = "Torch")
	ETorchModelState GetModelState(const UTorchModel* Model) const;

	UFUNCTION(BlueprintPure, Category = "Torch")
	bool IsModelReady(const UTorchModel* Model) const { return GetModelState(Model) == ETorchModelState::Ready; }

	/** The device the model was loaded on ("cuda", "cpu", ...), once loaded. */
	UFUNCTION(BlueprintPure, Category = "Torch")
	FString GetModelDevice(const UTorchModel* Model) const;

	/** The service running the model's script (created and started on first use). */
	UPythonService* GetService(const UTorchModel* Model);

	/** Fill the asset's Methods and Jobs from the script. */
	void FetchSignature(UTorchModel* Model, TFunction<void(bool bOk, const FString& Error)> OnDone);

	/** Calls, rows, forward passes, largest batch and forward time so far (JSON), optionally resetting them. */
	void GetStats(UTorchModel* Model, TFunction<void(const FPythonCallResult&)> OnResult, bool bReset = false);

	/** Start a @ut.job of the model's script. ParamsJson: a JSON object of the job's parameters. bResume continues
	 the job's latest run (Saved/Torch/Runs/<job>/). bOwnProcess runs it in a Python process of its own (also
	 when the script declares @ut.job(process="dedicated") and the asset's signature is refreshed); its
	 job.publish() then reaches the model through a weights file. Blueprint: Start Job. */
	UFUNCTION(BlueprintCallable, Category = "Torch", meta = (AutoCreateRefTerm = "ParamsJson"))
	UPythonJob* StartJob(UTorchModel* Model, const FString& JobName, const FString& ParamsJson, bool bResume, bool bOwnProcess = false);

	/** Where job runs go: <Project>/Saved/Torch/Runs. */
	static FString RunsDir();

	/** Torch Output Components: their streams' results come back by binding id. */
	int32 RegisterBinding(UTorchOutputComponent* Component);
	void UnregisterBinding(int32 Id);

	/** A model reply's outputs for one request: Offset and Rows into the reply's rows; without the batch
	 dimension unless bBatched. */
	static FTorchResult MakeResult(const TSharedPtr<FJsonObject>& Reply, const TArray<FPyTensor>& Tensors, int64 Offset, int32 Rows, bool bBatched);

	UPROPERTY(BlueprintAssignable, Category = "Torch")
	FTorchModelLoadedSignature OnModelLoaded;

	/** "Script|ModelName|Process": what identifies a loaded model. */
	static FString ModelKey(const UTorchModel* Model);

	/** Whether this build runs Python (packaged builds with Python CLI > Packaged Env Mode: None don't). */
	UFUNCTION(BlueprintPure, Category = "Torch")
	static bool IsPythonAvailable();

	/** Whether the method's requests run on the model's NNE export: Backend NNE, or Auto without Python. */
	bool UsesNNE(const UTorchModel* Model, const FString& Method) const;

#if WITH_EDITOR
	/** Export the model's methods to ONNX (Python), import them as NNE model data next to the asset and check
	 them against torch; fills the asset's NNE Methods. OnDone: success and a summary. */
	void ExportToNNE(UTorchModel* Model, TFunction<void(bool bOk, const FString& Summary)> OnDone);
#endif

private:
	struct FModelEntry
	{
		ETorchModelState State = ETorchModelState::NotLoaded;
		FString Error;
		FString Device;
		double LoadMs = 0.0;
		TArray<TFunction<void(bool, const FString&)>> Waiters;
	};

	struct FRequest
	{
		TWeakObjectPtr<UTorchModel> Model;
		FString Method;
		TMap<FString, FPyTensor> Inputs;
		FString JsonInputs;
		FTorchRunOptions Options;
		TFunction<void(const FTorchResult&)> OnResult;
		double Start = 0.0;
		int32 Rows = 1;
	};

	void EnsureLoadSent(UTorchModel* Model);
	void SendGroup(UTorchModel* Model, const FString& Method, TArray<FRequest>&& Requests);
	void SendGroupNNE(UTorchModel* Model, const FString& Method, TArray<FRequest>&& Requests);
	/** The requests' tensor inputs stacked on dim 0 (sorted by name), and the total rows. */
	static void StackInputs(const TArray<FRequest>& Requests, TArray<FString>& OutNames, TArray<FPyTensor>& OutTensors, int64& OutRows);
	/** Whether the model as a whole runs on NNE (every request would). */
	bool ModelUsesNNE(const UTorchModel* Model) const;
#if WITH_EDITOR
	void ExportMethod(TWeakObjectPtr<UTorchModel> Model, TArray<FString> Methods, int32 Index, TSharedRef<TArray<FString>> Summary, TSharedRef<bool> bAllOk,
		TFunction<void(bool, const FString&)> OnDone);
	bool ImportExport(UTorchModel* Model, const TSharedPtr<FJsonObject>& Desc, FString& OutSummary);
#endif
	static void Fail(TArray<FRequest>& Requests, const FString& Error);
	void HandleEndFrame();
	void BindService(UPythonService* Service);
	void ResetStoppedModels();
	void LoadStartupModels();
	/** A job in its own process published weights: have the model's process load them. */
	void ForwardPublish(UTorchModel* Model, const FPythonEventData& Event);
	/** A stream frame's result (a Torch Output Component's binding). */
	void HandleStreamResult(const FPythonEventData& Event);

	UFUNCTION()
	void HandleServiceStopped(const FString& Script);
	UFUNCTION()
	void HandleServiceError(const FString& Message, const FString& Traceback);

	TMap<FString, FModelEntry> Models;
	TMap<FString, TArray<FRequest>> Pending;
	TArray<TWeakObjectPtr<UPythonService>> BoundServices;
	TMap<int32, TWeakObjectPtr<UTorchOutputComponent>> Bindings;
	int32 NextBindingId = 0;
	FDelegateHandle EndFrameHandle;
	FDelegateHandle PostEngineInitHandle;
};
