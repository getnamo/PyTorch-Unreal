#include "TorchComponent.h"
#include "PyTorch.h"
#include "PythonJob.h"
#include "TorchBlueprintLibrary.h"
#include "TorchModel.h"
#include "TorchSubsystem.h"
#include "Dom/JsonObject.h"

UTorchComponent::UTorchComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
}

void UTorchComponent::BeginPlay()
{
	Super::BeginPlay();
	UTorchSubsystem* Torch = UTorchSubsystem::Get();
	if (bLoadOnBeginPlay && Model && Torch)
	{
		TWeakObjectPtr<UTorchComponent> Weak(this);
		Torch->LoadModel(Model, [Weak](bool bOk, const FString& Error)
		{
			if (UTorchComponent* Self = Weak.Get())
			{
				Self->OnModelLoaded.Broadcast(bOk, Error);
			}
		});
	}
}

void UTorchComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	if (bStopJobsOnEndPlay)
	{
		StopJobs();
	}
	Super::EndPlay(EndPlayReason);
}

bool UTorchComponent::IsModelReady() const
{
	const UTorchSubsystem* Torch = UTorchSubsystem::Get();
	return Torch && Model && Torch->IsModelReady(Model);
}

int32 UTorchComponent::Run(const FString& Method, const TMap<FString, FPyTensor>& Inputs, const FString& JsonInputs)
{
	const int32 Id = ++NextRequestId;
	TWeakObjectPtr<UTorchComponent> Weak(this);
	auto Deliver = [Weak, Id, Method](const FTorchResult& Result)
	{
		if (UTorchComponent* Self = Weak.Get())
		{
			if (!Result.bSuccess)
			{
				UE_LOG(LogTorch, Warning, TEXT("%s: %s failed: %s"), *Self->GetPathName(), *Method, *Result.Error);
			}
			Self->OnResult.Broadcast(Id, Method, Result);
		}
	};
	if (UTorchSubsystem* Torch = UTorchSubsystem::Get())
	{
		Torch->Run(Model, Method, Inputs, JsonInputs, Deliver);
	}
	else
	{
		FTorchResult R;
		R.Error = TEXT("the Torch subsystem isn't available");
		Deliver(R);
	}
	return Id;
}

int32 UTorchComponent::RunWithTensor(const FString& Method, const FString& InputName, const FPyTensor& Tensor, const FString& JsonInputs)
{
	TMap<FString, FPyTensor> Inputs;
	if (!InputName.IsEmpty())
	{
		Inputs.Add(InputName, Tensor);
	}
	return Run(Method, Inputs, JsonInputs);
}

int32 UTorchComponent::RunWithRenderTarget(const FString& Method, const FString& InputName, UTextureRenderTarget2D* RenderTarget, const FString& JsonInputs)
{
	return RunWithTensor(Method, InputName, UTorchBlueprintLibrary::MakeImageTensorFromRenderTarget(RenderTarget), JsonInputs);
}

UPythonJob* UTorchComponent::StartJob(const FString& JobName, const FString& ParamsJson, bool bResume, bool bOwnProcess)
{
	UTorchSubsystem* Torch = UTorchSubsystem::Get();
	if (!Torch)
	{
		return nullptr;
	}
	UPythonJob* Job = Torch->StartJob(Model, JobName, ParamsJson, bResume, bOwnProcess);
	LastJob = Job;
	Jobs.RemoveAll([](const TObjectPtr<UPythonJob>& J) { return !J || J->IsFinished(); });
	if (Job->IsFinished()) //Failed at once (no model, bad parameters)
	{
		HandleJobFailed(Job, Job->Error, Job->Traceback);
		return Job;
	}
	Jobs.Add(Job);
	Job->OnProgress.AddDynamic(this, &UTorchComponent::HandleJobProgress);
	Job->OnMetrics.AddDynamic(this, &UTorchComponent::HandleJobMetrics);
	Job->OnFinished.AddDynamic(this, &UTorchComponent::HandleJobFinished);
	Job->OnFailed.AddDynamic(this, &UTorchComponent::HandleJobFailed);
	return Job;
}

void UTorchComponent::StopJobs()
{
	for (UPythonJob* Job : Jobs)
	{
		if (Job && !Job->IsFinished())
		{
			Job->Stop();
		}
	}
}

bool UTorchComponent::IsJobRunning() const
{
	for (const UPythonJob* Job : Jobs)
	{
		if (Job && !Job->IsFinished())
		{
			return true;
		}
	}
	return false;
}

void UTorchComponent::HandleJobProgress(UPythonJob* Job, float Progress, const FString& Message)
{
	OnJobProgress.Broadcast(Job, Progress, Message);
}

void UTorchComponent::HandleJobMetrics(UPythonJob* Job, int32, const FString& MetricsJson)
{
	OnJobMetrics.Broadcast(Job, MetricsJson);
}

void UTorchComponent::HandleJobFinished(UPythonJob* Job, const FString& ResultJson)
{
	FString Text = ResultJson;
	const TSharedPtr<FJsonObject>* Obj = nullptr;
	if (Job && Job->Result.IsValid() && Job->Result->TryGetObject(Obj))
	{
		(*Obj)->TryGetStringField(TEXT("text"), Text);
	}
	OnJobFinished.Broadcast(Job, Text);
}

void UTorchComponent::HandleJobFailed(UPythonJob* Job, const FString& Error, const FString&)
{
	OnJobFailed.Broadcast(Job, Error);
}
