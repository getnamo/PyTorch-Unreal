// PyTorch.Packaged.*: the example models as a shipped game runs them (also in the editor). Needs the examples'
// NNE exports (PyTorch.Tools.ExportExamples) and /PyTorch/Examples cooked (packaged builds: Directories To Always
// Cook).
// - NNE: DA_Digits classifies a drawn 1 and the reach policy drives the agents, on NNE (no Python call: forced
//   in the editor, Auto in builds without Python).
// - Python: the same through Python (packaged builds that ship Python; skipped without it).
// PyTorch.Tools.ExportExamples (editor, on demand): Export To NNE on DA_Digits and DA_ReachPolicy.

#include "CoreMinimal.h"
#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS

#include "TorchBlueprintLibrary.h"
#include "TorchModel.h"
#include "TorchReachExample.h"
#include "TorchSubsystem.h"
#include "TorchTestHelpers.h"

using namespace TorchTests;

namespace TorchPackagedTests
{
	static UTorchModel* Load(const TCHAR* Path)
	{
		return LoadObject<UTorchModel>(nullptr, Path);
	}

	/** A 28x28 gray "1": a vertical bar, light on dark like MNIST. */
	static FPyTensor DrawnOne()
	{
		TArray<float> Pixels;
		Pixels.SetNumZeroed(28 * 28);
		for (int32 y = 5; y < 23; ++y)
		{
			for (int32 x = 13; x < 16; ++x)
			{
				Pixels[y * 28 + x] = 1.f;
			}
		}
		return FPyTensor::FromFloats(Pixels, { 28, 28, 1 });
	}

	struct FRunState
	{
		FEnv Env;
		UTorchModel* Digits = nullptr;
		UTorchModel* Policy = nullptr;
		ATorchReachEnvironment* E = nullptr;
		TSharedPtr<FTorchResult> Classify, ClassifyGpu;
		ETorchNNERuntime DigitsRuntime = ETorchNNERuntime::Auto;
		int32 EpisodesFrom = 0;
		ETorchBackend DigitsBackend = ETorchBackend::Auto, PolicyBackend = ETorchBackend::Auto;
	};

	/** Classify the drawn 1, then run the reach agents for 100 episodes, with the models on Backend. */
	static void RunExamples(FAutomationTestBase* Test, ETorchBackend Backend, const FString& Expected)
	{
		const TSharedRef<FRunState> S = MakeShared<FRunState>();
		S->Digits = Load(TEXT("/PyTorch/Examples/DA_Digits.DA_Digits"));
		S->Policy = Load(TEXT("/PyTorch/Examples/DA_ReachPolicy.DA_ReachPolicy"));
		if (!Test->TestNotNull(TEXT("DA_Digits (cook /PyTorch/Examples)"), S->Digits) || !Test->TestNotNull(TEXT("DA_ReachPolicy"), S->Policy))
		{
			return;
		}
		if (Backend == ETorchBackend::NNE && (!S->Digits->FindNNEMethod(TEXT("classify")) || !S->Policy->FindNNEMethod(TEXT("act"))))
		{
			Test->AddError(TEXT("the examples have no NNE exports: run PyTorch.Tools.ExportExamples in the editor"));
			return;
		}
		S->DigitsBackend = S->Digits->Backend;
		S->PolicyBackend = S->Policy->Backend;
		S->Digits->Backend = Backend;
		S->Policy->Backend = Backend;
		S->Classify = MakeShared<FTorchResult>();
		UTorchSubsystem::Get()->Run(S->Digits, TEXT("classify"), { { TEXT("image"), DrawnOne() } }, FString(), [Out = S->Classify](const FTorchResult& R) { *Out = R; });
		WaitUntil(Test, TEXT("classify"), [S] { return S->Classify->bSuccess || !S->Classify->Error.IsEmpty(); });
		Step([Test, S, Expected]
		{
			const FTorchResult& R = *S->Classify;
			Test->TestTrue(TEXT("classify: ") + R.Error, R.bSuccess);
			Test->AddInfo(FString::Printf(TEXT("drawn 1 -> %d (%.2f) on %s, %.3f ms"), UTorchBlueprintLibrary::GetOutputInt(R, TEXT("digit")),
				UTorchBlueprintLibrary::GetOutputFloat(R, TEXT("confidence")), *R.Backend, R.ModelMs));
			Test->TestEqual(TEXT("the drawn 1 reads as 1"), UTorchBlueprintLibrary::GetOutputInt(R, TEXT("digit")), 1);
			Test->TestTrue(FString::Printf(TEXT("on %s (%s)"), *Expected, *R.Backend), R.Backend.StartsWith(Expected));
			S->ClassifyGpu = MakeShared<FTorchResult>();
			if (Expected == TEXT("nne:"))
			{
				//The GPU runtime too (DirectML on Win64): Auto keeps the CPU for one small image
				S->DigitsRuntime = S->Digits->NNERuntime;
				S->Digits->NNERuntime = ETorchNNERuntime::GPU;
				UTorchSubsystem::Get()->Run(S->Digits, TEXT("classify"), { { TEXT("image"), DrawnOne() } }, FString(), [Out = S->ClassifyGpu](const FTorchResult& G) { *Out = G; });
			}
			else
			{
				S->ClassifyGpu->bSuccess = true;
			}
		});
		WaitUntil(Test, TEXT("classify on the GPU runtime"), [S] { return S->ClassifyGpu->bSuccess || !S->ClassifyGpu->Error.IsEmpty(); });
		Step([Test, S, Expected]
		{
			if (Expected == TEXT("nne:"))
			{
				const FTorchResult& G = *S->ClassifyGpu;
				Test->AddInfo(FString::Printf(TEXT("drawn 1 -> %d on %s, %.3f ms"), UTorchBlueprintLibrary::GetOutputInt(G, TEXT("digit")), *G.Backend, G.ModelMs));
				Test->TestTrue(TEXT("classify on the GPU runtime: ") + G.Error, G.bSuccess && UTorchBlueprintLibrary::GetOutputInt(G, TEXT("digit")) == 1);
				Test->TestEqual(TEXT("on DirectML"), G.Backend, FString(TEXT("nne:NNERuntimeORTDml")));
				S->Digits->NNERuntime = S->DigitsRuntime;
			}
			S->Env.Create();
			ATorchReachEnvironment* E = S->Env.World->SpawnActorDeferred<ATorchReachEnvironment>(ATorchReachEnvironment::StaticClass(), FTransform::Identity);
			E->NumAgents = 32;
			E->Mode = ETorchEnvironmentMode::Run;
			E->PolicyModel = S->Policy;
			E->Seed = 11;
			E->bExampleControls = false;
			E->FinishSpawning(FTransform::Identity);
			E->DispatchBeginPlay();
			S->Env.Actors.Add(E);
			S->E = E;
		});
		WaitUntil(Test, TEXT("the reach policy"), [S]
		{
			for (int32 i = 0; i < 4 && S->E->StepEnvironment(); ++i)
			{
			}
			int64 Steps;
			int32 Episodes;
			float Mean;
			S->E->GetStats(Steps, Episodes, Mean);
			return Episodes >= 100 || S->E->Mode != ETorchEnvironmentMode::Run;
		}, 300.0);
		Step([Test, S]
		{
			int64 Steps;
			int32 Episodes;
			float Mean;
			S->E->GetStats(Steps, Episodes, Mean);
			Test->AddInfo(FString::Printf(TEXT("reach policy: %d episodes in %lld steps, mean return %.2f"), Episodes, Steps, Mean));
			Test->TestEqual(TEXT("still in Run mode (the policy didn't fail)"), S->E->Mode, ETorchEnvironmentMode::Run);
			Test->TestTrue(FString::Printf(TEXT("the shipped policy reaches its targets (mean return %.2f)"), Mean), Mean > 1.0f);
			S->Env.Destroy();
			S->Digits->Backend = S->DigitsBackend;
			S->Policy->Backend = S->PolicyBackend;
		});
	}
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTorchPackagedNNETest, "PyTorch.Packaged.NNE", TorchTests::Flags)
bool FTorchPackagedNNETest::RunTest(const FString&)
{
	AddInfo(FString::Printf(TEXT("Python in this build: %s"), UTorchSubsystem::IsPythonAvailable() ? TEXT("yes") : TEXT("no")));
	//Without Python, Auto is NNE: what a game packaged without Python does
	TorchPackagedTests::RunExamples(this, UTorchSubsystem::IsPythonAvailable() ? ETorchBackend::NNE : ETorchBackend::Auto, TEXT("nne:"));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTorchPackagedPythonTest, "PyTorch.Packaged.Python", TorchTests::Flags)
bool FTorchPackagedPythonTest::RunTest(const FString&)
{
	if (!UTorchSubsystem::IsPythonAvailable())
	{
		AddInfo(TEXT("this build has no Python (Packaged Env Mode: None): skipped"));
		return true;
	}
	TorchPackagedTests::RunExamples(this, ETorchBackend::Python, TEXT("python"));
	return true;
}

#if WITH_EDITOR
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTorchExportExamplesTool, "PyTorch.Tools.ExportExamples", TorchTests::BenchmarkFlags)
bool FTorchExportExamplesTool::RunTest(const FString&)
{
	struct FState
	{
		TArray<UTorchModel*> Models;
		int32 Done = 0;
	};
	const TSharedRef<FState> S = MakeShared<FState>();
	S->Models = { TorchPackagedTests::Load(TEXT("/PyTorch/Examples/DA_Digits.DA_Digits")), TorchPackagedTests::Load(TEXT("/PyTorch/Examples/DA_ReachPolicy.DA_ReachPolicy")) };
	if (!TestTrue(TEXT("the example models"), S->Models[0] && S->Models[1]))
	{
		return true;
	}
	S->Models[0]->ExportMethods = { TEXT("classify") };
	S->Models[1]->ExportMethods = { TEXT("act") };
	Step([this, S]
	{
		for (UTorchModel* M : S->Models)
		{
			UTorchSubsystem::Get()->ExportToNNE(M, [this, S, Name = M->GetName()](bool bOk, const FString& Summary)
			{
				AddInfo(FString::Printf(TEXT("%s: %s"), *Name, *Summary));
				TestTrue(Name + TEXT(" exported"), bOk);
				++S->Done;
			});
		}
	});
	WaitUntil(this, TEXT("exports"), [S] { return S->Done == S->Models.Num(); }, 600.0);
	return true;
}
#endif

#endif
