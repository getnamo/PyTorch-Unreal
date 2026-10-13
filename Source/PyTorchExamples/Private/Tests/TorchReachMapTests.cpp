// Plays the reach example (/PyTorch/Examples/ReachExample) in PIE:
// 1. Training (PPO, started on BeginPlay) until the environment hands over to the trained policy (Run).
// 2. The policy's mean return over 100 episodes; a screenshot lands in Saved/Screenshots.
// 3. [C] records the scripted expert, [B] clones it (behavior cloning), [R] runs the clone.
// Needs a real RHI (-RenderOffscreen) and the editor.

#include "CoreMinimal.h"
#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS && WITH_EDITOR

#include "PythonJob.h"
#include "TorchReachExample.h"
#include "DynamicRHI.h"
#include "Editor.h"
#include "EngineUtils.h"
#include "Engine/World.h"
#include "GameFramework/PlayerController.h"
#include "InputCoreTypes.h"
#include "InputKeyEventArgs.h"
#include "Tests/AutomationCommon.h"
#include "Tests/AutomationEditorCommon.h"
#include "UnrealClient.h"

namespace TorchReachMapTests
{
	enum class EPhase : uint8 { Training, Running, Recording, Cloning, RunningClone, Done };

	struct FState
	{
		double Deadline = 0.0;
		TWeakObjectPtr<ATorchReachEnvironment> Env;
		EPhase Phase = EPhase::Training;
		int32 EpisodesFrom = 0;
		int64 StepsFrom = 0;
		bool bScreenshot = false;
		double ScreenshotAt = 0.0;
	};

	static void Press(UWorld* World, const FKey& Key)
	{
		if (APlayerController* PC = World->GetFirstPlayerController())
		{
			const FInputDeviceId Device = FInputDeviceId::CreateFromInternalId(0);
			PC->InputKey(FInputKeyEventArgs(nullptr, Device, Key, IE_Pressed, FPlatformTime::Cycles64()));
			PC->InputKey(FInputKeyEventArgs(nullptr, Device, Key, IE_Released, FPlatformTime::Cycles64()));
		}
	}

	static void Stats(const ATorchEnvironment* E, int64& Steps, int32& Episodes, float& Mean)
	{
		E->GetStats(Steps, Episodes, Mean);
	}
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTorchReachMapTest, "PyTorch.Examples.ReachMap", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FTorchReachMapTest::RunTest(const FString&)
{
	using namespace TorchReachMapTests;
	if (!FApp::CanEverRender() || !GDynamicRHI || FCString::Strcmp(GDynamicRHI->GetName(), TEXT("Null")) == 0)
	{
		AddWarning(TEXT("needs a real RHI (run with -RenderOffscreen instead of -nullrhi): skipped"));
		return true;
	}
	TSharedRef<FState> S = MakeShared<FState>();
	ADD_LATENT_AUTOMATION_COMMAND(FEditorLoadMap(TEXT("/PyTorch/Examples/ReachExample")));
	ADD_LATENT_AUTOMATION_COMMAND(FStartPIECommand(false));
	ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand([this, S]() -> bool
	{
		const double Now = FPlatformTime::Seconds();
		if (S->Deadline == 0.0)
		{
			S->Deadline = Now + 600.0;
		}
		if (Now > S->Deadline)
		{
			AddError(FString::Printf(TEXT("timed out in phase %d: %s"), (int32)S->Phase, S->Env.IsValid() ? *S->Env->GetStatus() : TEXT("no environment")));
			return true;
		}
		UWorld* World = GEditor ? GEditor->PlayWorld.Get() : nullptr;
		if (!World)
		{
			return false;
		}
		if (!S->Env.IsValid())
		{
			for (TActorIterator<ATorchReachEnvironment> It(World); It; ++It)
			{
				S->Env = *It;
			}
			return false;
		}
		ATorchReachEnvironment* E = S->Env.Get();
		int64 Steps;
		int32 Episodes;
		float Mean;
		Stats(E, Steps, Episodes, Mean);
		switch (S->Phase)
		{
		case EPhase::Training:
			if (E->GetTrainingJob() && E->GetTrainingJob()->State == EPythonJobState::Failed)
			{
				AddError(TEXT("training failed: ") + E->GetTrainingJob()->Error);
				return true;
			}
			if (E->Mode == ETorchEnvironmentMode::Run)
			{
				AddInfo(TEXT("trained: ") + E->GetStatus().Replace(TEXT("\n"), TEXT(" | ")));
				S->Phase = EPhase::Running;
				S->EpisodesFrom = Episodes;
			}
			return false;
		case EPhase::Running:
			if (Episodes < S->EpisodesFrom + 100)
			{
				return false;
			}
			if (!S->bScreenshot)
			{
				S->bScreenshot = true;
				S->ScreenshotAt = Now;
				FScreenshotRequest::RequestScreenshot(TEXT("PyTorch_ReachExample.png"), false, false);
				return false;
			}
			if (Now - S->ScreenshotAt < 1.0)
			{
				return false;
			}
			AddInfo(FString::Printf(TEXT("trained policy in PIE: mean return %.2f"), Mean));
			TestTrue(FString::Printf(TEXT("the trained policy reaches its targets (mean return %.2f)"), Mean), Mean > 1.2f);
			TestTrue(TEXT("the status text shows Run: ") + E->GetStatus(), E->GetStatus().StartsWith(TEXT("Reach: Run")));
			Press(World, EKeys::C);
			S->Phase = EPhase::Recording;
			S->StepsFrom = Steps;
			return false;
		case EPhase::Recording:
			if (Steps < S->StepsFrom + 200)
			{
				return false;
			}
			TestEqual(TEXT("[C]: recording"), E->Mode, ETorchEnvironmentMode::Record);
			Press(World, EKeys::B);
			S->Phase = EPhase::Cloning;
			return false;
		case EPhase::Cloning:
		{
			const FString Status = E->GetStatus();
			if (!Status.Contains(TEXT("train_reach_bc: Cloned")) && !Status.Contains(TEXT("train_reach_bc: Failed")))
			{
				return false;
			}
			AddInfo(TEXT("cloned: ") + Status.Replace(TEXT("\n"), TEXT(" | ")));
			TestTrue(TEXT("[B]: behavior cloning from the recording"), Status.Contains(TEXT("train_reach_bc: Cloned")));
			Press(World, EKeys::R);
			S->Phase = EPhase::RunningClone;
			S->EpisodesFrom = Episodes;
			return false;
		}
		case EPhase::RunningClone:
			if (Episodes < S->EpisodesFrom + 100)
			{
				return false;
			}
			AddInfo(FString::Printf(TEXT("cloned policy in PIE: mean return %.2f"), Mean));
			TestTrue(FString::Printf(TEXT("[R]: the clone reaches its targets (mean return %.2f)"), Mean), Mean > 1.2f);
			S->Phase = EPhase::Done;
			return true;
		default:
			return true;
		}
	}));
	ADD_LATENT_AUTOMATION_COMMAND(FEndPlayMapCommand());
	return true;
}

#endif
