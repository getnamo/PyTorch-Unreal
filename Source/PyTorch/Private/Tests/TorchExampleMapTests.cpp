// Plays the example map (/PyTorch/Examples/MnistExample) in PIE. Each digit drawn into the canvas is read by the
// live model; training publishes weights as it goes.
// 1. Wait for the training started on BeginPlay (or its resume) to finish.
// 2. Press T: a fresh run from random weights. Readings start wrong and become right.
// 3. Press S after the first epoch: stopped.
// 4. Press R: resumed to the end.
// 5. Press N: the next digit.
// Needs a real RHI (-RenderOffscreen) and the editor; a screenshot lands in Saved/Screenshots.

#include "CoreMinimal.h"
#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS && WITH_EDITOR

#include "TorchComponent.h"
#include "Components/TextRenderComponent.h"
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

namespace TorchExampleMapTests
{
	enum class EPhase : uint8 { WaitStartup, Fresh, Stopping, Resuming, Matching, NextDigit, Done };

	struct FState
	{
		double Start = 0.0;
		double Deadline = 0.0;
		TWeakObjectPtr<AActor> Example;
		FString LastStatus;
		TArray<FString> StatusLines;
		EPhase Phase = EPhase::WaitStartup;
		double PhaseStart = 0.0;
		FString LastSeen;
		int32 FreshRight = 0, FreshWrong = 0, FreshFirstWrong = -1, FreshLastRight = -1, Reads = 0;
		bool bFreshStarted = false;
		FString DigitBeforeKey;
		FString Matched;
	};

	static FString TextOf(AActor* Actor, const TCHAR* ComponentName)
	{
		for (UActorComponent* C : Actor->GetComponents())
		{
			if (UTextRenderComponent* T = Cast<UTextRenderComponent>(C); T && T->GetName() == ComponentName)
			{
				return T->Text.ToString();
			}
		}
		return FString();
	}

	/** "Drawn: test image 123, a 7" -> 7; "Camera sees: 7 (98%)" -> 7. */
	static int32 DigitIn(const FString& Text, const TCHAR* Marker)
	{
		const int32 At = Text.Find(Marker);
		if (At == INDEX_NONE || At + FCString::Strlen(Marker) >= Text.Len())
		{
			return -1;
		}
		const TCHAR C = Text[At + FCString::Strlen(Marker)];
		return FChar::IsDigit(C) ? C - TEXT('0') : -1;
	}

	static void Press(UWorld* World, const FKey& Key)
	{
		if (APlayerController* PC = World->GetFirstPlayerController())
		{
			const FInputDeviceId Device = FInputDeviceId::CreateFromInternalId(0);
			PC->InputKey(FInputKeyEventArgs(nullptr, Device, Key, IE_Pressed, FPlatformTime::Cycles64()));
			PC->InputKey(FInputKeyEventArgs(nullptr, Device, Key, IE_Released, FPlatformTime::Cycles64()));
		}
	}
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTorchExampleMapTest, "PyTorch.Examples.MnistMap", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FTorchExampleMapTest::RunTest(const FString&)
{
	using namespace TorchExampleMapTests;
	if (!FApp::CanEverRender() || !GDynamicRHI || FCString::Strcmp(GDynamicRHI->GetName(), TEXT("Null")) == 0)
	{
		AddWarning(TEXT("needs a real RHI (run with -RenderOffscreen instead of -nullrhi): skipped"));
		return true;
	}
	TSharedRef<FState> S = MakeShared<FState>();
	ADD_LATENT_AUTOMATION_COMMAND(FEditorLoadMap(TEXT("/PyTorch/Examples/MnistExample")));
	ADD_LATENT_AUTOMATION_COMMAND(FStartPIECommand(false));
	ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand([this, S]() -> bool
	{
		const double Now = FPlatformTime::Seconds();
		if (S->Start == 0.0)
		{
			S->Start = Now;
			S->Deadline = Now + 900.0; //First run: environment, MNIST download, training
		}
		UWorld* World = GEditor ? GEditor->PlayWorld.Get() : nullptr;
		if (!World)
		{
			return Now > S->Deadline;
		}
		if (!S->Example.IsValid())
		{
			for (TActorIterator<AActor> It(World); It; ++It)
			{
				if (It->GetComponentByClass<UTorchComponent>())
				{
					S->Example = *It;
				}
			}
			return false;
		}
		AActor* Example = S->Example.Get();
		const FString Status = TextOf(Example, TEXT("StatusText"));
		if (Status != S->LastStatus)
		{
			S->LastStatus = Status;
			S->StatusLines.Add(Status);
			UE_LOG(LogTemp, Display, TEXT("MnistMap status: %s"), *Status);
		}
		const FString Digit = TextOf(Example, TEXT("DigitText"));
		const FString Seen = TextOf(Example, TEXT("PredictionText"));
		const int32 Drawn = DigitIn(Digit, TEXT(", a "));
		const int32 Read = DigitIn(Seen, TEXT("Camera sees: "));
		const bool bTrained = Status.StartsWith(TEXT("Trained"));
		auto Next = [S, Now](EPhase Phase) { S->Phase = Phase; S->PhaseStart = Now; };

		if (Now > S->Deadline)
		{
			AddError(FString::Printf(TEXT("stuck in phase %d; status: %s; digit: %s; prediction: %s"), (int32)S->Phase, *Status, *Digit, *Seen));
			return true;
		}
		switch (S->Phase)
		{
		case EPhase::WaitStartup:
			if (Status.StartsWith(TEXT("Training failed")))
			{
				AddError(Status);
				return true;
			}
			if (bTrained || Status.StartsWith(TEXT("Stopped")))
			{
				Press(World, EKeys::T); //A fresh run: random weights, then better and better
				Next(EPhase::Fresh);
			}
			return false;
		case EPhase::Fresh:
			if (!S->bFreshStarted)
			{
				//Until the new run reports, the status still shows the previous one
				S->bFreshStarted = Status.StartsWith(TEXT("Training")) || Status.StartsWith(TEXT("Loading"));
				return false;
			}
			if (Seen != S->LastSeen && Drawn >= 0 && Read >= 0 && Status.StartsWith(TEXT("Training")))
			{
				S->LastSeen = Seen;
				const bool bRight = Drawn == Read;
				(bRight ? S->FreshRight : S->FreshWrong)++;
				if (!bRight && S->FreshFirstWrong < 0)
				{
					S->FreshFirstWrong = S->Reads;
				}
				if (bRight)
				{
					S->FreshLastRight = S->Reads;
				}
				++S->Reads;
			}
			if (Status.StartsWith(TEXT("Epoch 1/2 done")) || Status.Contains(TEXT("epoch 2/2"))) //A status can last a single frame
			{
				Press(World, EKeys::S);
				Next(EPhase::Stopping);
			}
			else if (bTrained)
			{
				AddWarning(TEXT("the fresh run finished before it could be stopped"));
				Next(EPhase::Matching);
			}
			return false;
		case EPhase::Stopping:
			if (Status.StartsWith(TEXT("Stopped")))
			{
				Press(World, EKeys::R);
				Next(EPhase::Resuming);
			}
			return false;
		case EPhase::Resuming:
			if (bTrained)
			{
				Next(EPhase::Matching);
			}
			return false;
		case EPhase::Matching:
			if (Drawn >= 0 && Drawn == Read)
			{
				S->Matched = FString::Printf(TEXT("%s | %s"), *Digit, *Seen);
				S->DigitBeforeKey = Digit;
				FScreenshotRequest::RequestScreenshot(TEXT("PyTorch_MnistExample.png"), false, false);
				Press(World, EKeys::N);
				Next(EPhase::NextDigit);
			}
			return false;
		case EPhase::NextDigit:
			if (Digit != S->DigitBeforeKey || Now - S->PhaseStart > 2.5)
			{
				TestTrue(TEXT("N drew another digit within 2.5 s"), Digit != S->DigitBeforeKey);
				Next(EPhase::Done);
			}
			return false;
		default:
			break;
		}
		TestTrue(TEXT("T trained a fresh run, S stopped it after an epoch, R resumed it to the end"),
			S->StatusLines.ContainsByPredicate([](const FString& L) { return L.StartsWith(TEXT("Stopped")); }));
		TestTrue(FString::Printf(TEXT("readings during the fresh run: %d right, %d wrong"), S->FreshRight, S->FreshWrong), S->FreshRight + S->FreshWrong > 0);
		AddInfo(FString::Printf(TEXT("after %.0f s: %s"), Now - S->Start, *S->Matched));
		AddInfo(FString::Printf(TEXT("fresh run readings: %d right, %d wrong (first wrong #%d, last right #%d)"), S->FreshRight, S->FreshWrong, S->FreshFirstWrong, S->FreshLastRight));
		AddInfo(TEXT("status lines: ") + FString::Join(S->StatusLines, TEXT(" / ")));
		return true;
	}));
	ADD_LATENT_AUTOMATION_COMMAND(FEndPlayMapCommand());
	return true;
}

#endif
