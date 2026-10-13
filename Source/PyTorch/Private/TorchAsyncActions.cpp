#include "TorchAsyncActions.h"
#include "PythonJob.h"
#include "TorchModel.h"
#include "TorchSubsystem.h"

UTorchRunModelAsyncAction* UTorchRunModelAsyncAction::RunModel(UTorchModel* InModel, const FString& InMethod, const TMap<FString, FPyTensor>& InInputs, const FString& InJsonInputs)
{
	UTorchRunModelAsyncAction* Action = NewObject<UTorchRunModelAsyncAction>();
	Action->Model = InModel;
	Action->Method = InMethod;
	Action->Inputs = InInputs;
	Action->JsonInputs = InJsonInputs;
	Action->AddToRoot(); //Models outlive worlds: kept alive until the result instead of by a game instance
	return Action;
}

UTorchRunModelAsyncAction* UTorchRunModelAsyncAction::RunModelWithFloats(UTorchModel* InModel, const FString& InMethod, const FString& InputName, const TArray<float>& Values, const TArray<int64>& Shape)
{
	TMap<FString, FPyTensor> In;
	In.Add(InputName, FPyTensor::FromFloats(Values, Shape));
	return RunModel(InModel, InMethod, In, FString());
}

void UTorchRunModelAsyncAction::Activate()
{
	UTorchSubsystem* Torch = UTorchSubsystem::Get();
	TWeakObjectPtr<UTorchRunModelAsyncAction> Weak(this);
	auto Finish = [Weak](const FTorchResult& Result)
	{
		if (UTorchRunModelAsyncAction* Self = Weak.Get())
		{
			(Result.bSuccess ? Self->OnSuccess : Self->OnFailure).Broadcast(Result);
			Self->RemoveFromRoot();
			Self->SetReadyToDestroy();
		}
	};
	if (!Torch)
	{
		FTorchResult R;
		R.Error = TEXT("Run Model: the Torch subsystem isn't available");
		Finish(R);
		return;
	}
	Torch->Run(Model.Get(), Method, Inputs, JsonInputs, Finish);
}

UTorchLoadModelAsyncAction* UTorchLoadModelAsyncAction::LoadTorchModel(UTorchModel* InModel)
{
	UTorchLoadModelAsyncAction* Action = NewObject<UTorchLoadModelAsyncAction>();
	Action->Model = InModel;
	Action->AddToRoot();
	return Action;
}

void UTorchLoadModelAsyncAction::Activate()
{
	UTorchSubsystem* Torch = UTorchSubsystem::Get();
	TWeakObjectPtr<UTorchLoadModelAsyncAction> Weak(this);
	auto Finish = [Weak](bool bOk, const FString& Error)
	{
		if (UTorchLoadModelAsyncAction* Self = Weak.Get())
		{
			(bOk ? Self->OnLoaded : Self->OnFailed).Broadcast(Error);
			Self->RemoveFromRoot();
			Self->SetReadyToDestroy();
		}
	};
	if (!Torch || !Model.IsValid())
	{
		Finish(false, TEXT("Load Model: no model"));
		return;
	}
	Torch->LoadModel(Model.Get(), Finish);
}

UTorchStartJobAsyncAction* UTorchStartJobAsyncAction::StartTorchJob(UTorchModel* InModel, const FString& InJobName, const FString& InParamsJson, bool bInResume, bool bInOwnProcess)
{
	UTorchStartJobAsyncAction* Action = NewObject<UTorchStartJobAsyncAction>();
	Action->Model = InModel;
	Action->JobName = InJobName;
	Action->ParamsJson = InParamsJson;
	Action->bResume = bInResume;
	Action->bOwnProcess = bInOwnProcess;
	Action->AddToRoot();
	return Action;
}

void UTorchStartJobAsyncAction::Activate()
{
	UTorchSubsystem* Torch = UTorchSubsystem::Get();
	if (!Torch)
	{
		Done();
		return;
	}
	Job = Torch->StartJob(Model.Get(), JobName, ParamsJson, bResume, bOwnProcess);
	Job->OnProgress.AddDynamic(this, &UTorchStartJobAsyncAction::HandleProgress);
	Job->OnMetrics.AddDynamic(this, &UTorchStartJobAsyncAction::HandleMetrics);
	Job->OnFinished.AddDynamic(this, &UTorchStartJobAsyncAction::HandleFinished);
	Job->OnFailed.AddDynamic(this, &UTorchStartJobAsyncAction::HandleFailed);
	OnStarted.Broadcast(Job);
	if (Job->State == EPythonJobState::Failed)
	{
		OnFailed.Broadcast(Job);
		Done();
	}
}

void UTorchStartJobAsyncAction::HandleProgress(UPythonJob* InJob, float, const FString&)
{
	OnProgress.Broadcast(InJob);
}

void UTorchStartJobAsyncAction::HandleMetrics(UPythonJob* InJob, int32, const FString&)
{
	OnMetrics.Broadcast(InJob);
}

void UTorchStartJobAsyncAction::HandleFinished(UPythonJob* InJob, const FString&)
{
	OnCompleted.Broadcast(InJob);
	Done();
}

void UTorchStartJobAsyncAction::HandleFailed(UPythonJob* InJob, const FString&, const FString&)
{
	OnFailed.Broadcast(InJob);
	Done();
}

void UTorchStartJobAsyncAction::Done()
{
	if (IsRooted())
	{
		RemoveFromRoot();
	}
	SetReadyToDestroy();
}
