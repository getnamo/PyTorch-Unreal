// PyTorch.Jobs.*: training jobs on examples/digits.py through UTorchComponent and UTorchSubsystem: progress,
// metrics, published weights, stop and resume, and a job in its own process. MNIST downloads on first use.

#include "CoreMinimal.h"
#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS

#include "PyTorch.h"
#include "PythonJob.h"
#include "TorchBlueprintLibrary.h"
#include "TorchComponent.h"
#include "TorchModel.h"
#include "TorchSubsystem.h"
#include "TorchTestHelpers.h"
#include "Dom/JsonObject.h"
#include "Misc/Paths.h"

// The tests are in a namespace of their own: a file-scope using-directive would leak into the other files of a unity build.
namespace TorchJobTestsSuite
{
using namespace TorchTests;

namespace TorchJobTests
{
	static UTorchModel* Digits()
	{
		static TWeakObjectPtr<UTorchModel> Model;
		if (!Model.IsValid())
		{
			UTorchModel* M = NewObject<UTorchModel>(GetTransientPackage(), TEXT("Test_JobsDigits"));
			M->Script = TEXT("examples/digits.py");
			M->ModelName = TEXT("digits");
			M->AddToRoot();
			Model = M;
		}
		return Model.Get();
	}

	struct FRun
	{
		bool bDone = false;
		FTorchResult Result;
	};

	static TSharedRef<FRun> Run(const FString& Method, const TMap<FString, FPyTensor>& Inputs, const FString& Json = FString())
	{
		TSharedRef<FRun> Out = MakeShared<FRun>();
		UTorchSubsystem::Get()->Run(Digits(), Method, Inputs, Json, [Out](const FTorchResult& R) { Out->Result = R; Out->bDone = true; });
		return Out;
	}
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTorchJobTrainTest, "PyTorch.Jobs.TrainDigits", TorchTests::Flags)
bool FTorchJobTrainTest::RunTest(const FString&)
{
	using namespace TorchJobTests;
	struct FState
	{
		FEnv Env;
		UTorchComponent* Torch = nullptr;
		UPythonJob* Job = nullptr;
		TSharedPtr<FRun> Sample, Read;
	};
	const TSharedRef<FState> S = MakeShared<FState>();
	S->Env.Create();
	S->Torch = S->Env.Add<UTorchComponent>(S->Env.Spawn(), [](UTorchComponent* C) { C->Model = Digits(); });
	Step([S]
	{
		S->Job = S->Torch->StartJob(TEXT("train_digits"), TEXT("{\"epochs\": 1, \"max_steps\": 150}"), false);
		S->Job->AddToRoot();
	});
	WaitUntil(this, TEXT("training"), [S] { return S->Job->IsFinished(); }, 600.0);
	Step([this, S]
	{
		UPythonJob* J = S->Job;
		TestEqual(TEXT("job completed: ") + J->Error, J->State, EPythonJobState::Completed);
		const float Accuracy = J->GetMetric(TEXT("accuracy"));
		TestTrue(FString::Printf(TEXT("one short epoch: %.1f%% test accuracy"), Accuracy * 100.f), Accuracy > 0.9f);
		TestTrue(TEXT("progress and its messages: ") + J->ProgressMessage, J->Progress == 1.f && J->ProgressMessage.StartsWith(TEXT("Epoch 1/1 done")));
		TestTrue(TEXT("run folder under Saved/Torch/Runs"), J->RunDir.Contains(TEXT("Torch")) && FPaths::FileExists(J->RunDir / TEXT("checkpoint.pt")));
		AddInfo(FString::Printf(TEXT("train_digits: %s (%s)"), *J->ResultJson, *J->RunDir));
		S->Sample = Run(TEXT("sample"), {}, TEXT("{\"index\": 7}"));
	});
	WaitUntil(this, TEXT("sample"), [S] { return S->Sample->bDone; });
	Step([this, S]
	{
		FPyTensor Image;
		TestTrue(TEXT("sample image: ") + S->Sample->Result.Error, UTorchBlueprintLibrary::GetOutputTensor(S->Sample->Result, TEXT("image"), Image) && Image.Shape == TArray<int64>({ 256, 256 }));
		S->Read = Run(TEXT("read"), { { TEXT("image"), Image } });
	});
	WaitUntil(this, TEXT("read"), [S] { return S->Read->bDone; });
	Step([this, S]
	{
		FString Text;
		UTorchBlueprintLibrary::GetOutputString(S->Read->Result, TEXT("text"), Text);
		TestTrue(TEXT("the live model (published weights) reads the 9: ") + Text, Text.StartsWith(TEXT("Camera sees: 9")));
		S->Job->RemoveFromRoot();
		S->Env.Destroy();
	});
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTorchJobResumeTest, "PyTorch.Jobs.StopAndResume", TorchTests::Flags)
bool FTorchJobResumeTest::RunTest(const FString&)
{
	using namespace TorchJobTests;
	struct FState
	{
		UPythonJob* First = nullptr;
		UPythonJob* Second = nullptr;
	};
	const TSharedRef<FState> S = MakeShared<FState>();
	Step([S]
	{
		S->First = UTorchSubsystem::Get()->StartJob(Digits(), TEXT("train_digits"), TEXT("{\"epochs\": 3, \"max_steps\": 300}"), false);
		S->First->AddToRoot();
	});
	//Stop it during its second epoch: the first epoch's checkpoint is what resume continues from
	WaitUntil(this, TEXT("an epoch done"), [S] { return S->First->IsFinished() || (S->First->LastMetrics.IsValid() && S->First->GetMetric(TEXT("epoch")) >= 1.f); }, 600.0);
	Step([S] { S->First->Stop(); });
	WaitUntil(this, TEXT("stopped"), [S] { return S->First->IsFinished(); }, 60.0);
	Step([this, S]
	{
		TestEqual(TEXT("stopped"), S->First->State, EPythonJobState::Stopped);
		S->Second = UTorchSubsystem::Get()->StartJob(Digits(), TEXT("train_digits"), TEXT("{\"epochs\": 3, \"max_steps\": 300}"), true);
		S->Second->AddToRoot();
	});
	WaitUntil(this, TEXT("resumed job"), [S] { return S->Second->IsFinished(); }, 600.0);
	Step([this, S]
	{
		TestEqual(TEXT("resumed job completed: ") + S->Second->Error, S->Second->State, EPythonJobState::Completed);
		TestTrue(TEXT("it continued the stopped run"), S->Second->bResumed && S->Second->RunDir == S->First->RunDir);
		TestTrue(TEXT("and finished the remaining epochs"), S->Second->GetMetric(TEXT("epoch")) == 3.f);
		S->First->RemoveFromRoot();
		S->Second->RemoveFromRoot();
	});
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTorchJobOwnProcessTest, "PyTorch.Jobs.OwnProcess", TorchTests::Flags)
bool FTorchJobOwnProcessTest::RunTest(const FString&)
{
	using namespace TorchJobTests;
	struct FState
	{
		UPythonJob* Job = nullptr;
		bool bLoaded = false;
		TSharedPtr<FPythonCallResult> Before, After;
	};
	const TSharedRef<FState> S = MakeShared<FState>();
	Step([S] { UTorchSubsystem::Get()->LoadModel(Digits(), [S](bool, const FString&) { S->bLoaded = true; }); });
	WaitUntil(this, TEXT("model loaded"), [S] { return S->bLoaded; });
	Step([S]
	{
		S->Before = MakeShared<FPythonCallResult>();
		UTorchSubsystem::Get()->GetStats(Digits(), [Out = S->Before](const FPythonCallResult& R) { *Out = R; });
		S->Job = UTorchSubsystem::Get()->StartJob(Digits(), TEXT("train_digits"), TEXT("{\"epochs\": 1, \"max_steps\": 60, \"publish_every\": 20}"), false, /*bOwnProcess*/ true);
		S->Job->AddToRoot();
	});
	WaitUntil(this, TEXT("job in its own process"), [S] { return S->Job->IsFinished(); }, 600.0);
	Step([S]
	{
		S->After = MakeShared<FPythonCallResult>();
		UTorchSubsystem::Get()->GetStats(Digits(), [Out = S->After](const FPythonCallResult& R) { *Out = R; });
	});
	WaitUntil(this, TEXT("stats"), [S] { return S->After->bSuccess || !S->After->Error.IsEmpty(); });
	Step([this, S]
	{
		TestEqual(TEXT("job completed: ") + S->Job->Error, S->Job->State, EPythonJobState::Completed);
		const double Before = S->Before->Result.IsValid() ? S->Before->Result->AsObject()->GetNumberField(TEXT("publishes")) : -1.0;
		const double After = S->After->Result.IsValid() ? S->After->Result->AsObject()->GetNumberField(TEXT("publishes")) : -1.0;
		TestTrue(FString::Printf(TEXT("its publish() reached the live model in the shared process (%.0f -> %.0f)"), Before, After), After > Before && Before >= 0.0);
		S->Job->RemoveFromRoot();
	});
	return true;
}

} // namespace TorchJobTestsSuite

#endif
