// Plays the vision example (/PyTorch/Examples/VisionExample) in PIE. A scene capture's render target streams to
// two Torch Output Components, with no scripts:
// - ImageNet classification, whose label is shown;
// - a segmentation overlay written into a second render target (image to image).
// Measures their rates (results per second) on the GPU. Needs a real RHI (-RenderOffscreen) and the editor;
// pretrained weights (35 MB) download once. A screenshot lands in Saved/Screenshots.

#include "CoreMinimal.h"
#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS && WITH_EDITOR

#include "PyTorch.h"
#include "PythonImageLibrary.h"
#include "PythonStreamComponent.h"
#include "TorchBlueprintLibrary.h"
#include "TorchOutputComponent.h"
#include "Components/TextRenderComponent.h"
#include "DynamicRHI.h"
#include "Editor.h"
#include "EngineUtils.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Engine/World.h"
#include "Tests/AutomationCommon.h"
#include "Tests/AutomationEditorCommon.h"
#include "UnrealClient.h"

namespace TorchVisionMapTests
{
	struct FState
	{
		double Start = 0.0;
		double Deadline = 0.0;
		double MeasureStart = 0.0;
		int32 ClassifyAtStart = 0, SegmentAtStart = 0;
		TWeakObjectPtr<AActor> Example;
		bool bScreenshot = false;
	};

	static UTorchOutputComponent* Output(AActor* Actor, const TCHAR* Name)
	{
		TArray<UTorchOutputComponent*> All;
		Actor->GetComponents(All);
		for (UTorchOutputComponent* C : All)
		{
			if (C->GetName() == Name)
			{
				return C;
			}
		}
		return nullptr;
	}
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTorchVisionMapTest, "PyTorch.Examples.VisionMap", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FTorchVisionMapTest::RunTest(const FString&)
{
	using namespace TorchVisionMapTests;
	if (!FApp::CanEverRender() || !GDynamicRHI || FCString::Strcmp(GDynamicRHI->GetName(), TEXT("Null")) == 0)
	{
		AddWarning(TEXT("needs a real RHI (run with -RenderOffscreen instead of -nullrhi): skipped"));
		return true;
	}
	TSharedRef<FState> S = MakeShared<FState>();
	ADD_LATENT_AUTOMATION_COMMAND(FEditorLoadMap(TEXT("/PyTorch/Examples/VisionExample")));
	ADD_LATENT_AUTOMATION_COMMAND(FStartPIECommand(false));
	ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand([this, S]() -> bool
	{
		const double Now = FPlatformTime::Seconds();
		if (S->Start == 0.0)
		{
			S->Start = Now;
			S->Deadline = Now + 600.0; //First run: torchvision weights download
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
				if (It->GetComponentByClass<UTorchOutputComponent>())
				{
					S->Example = *It;
				}
			}
			return false;
		}
		UTorchOutputComponent* Classifier = Output(S->Example.Get(), TEXT("Classifier"));
		UTorchOutputComponent* Segmenter = Output(S->Example.Get(), TEXT("Segmenter"));
		if (!Classifier || !Segmenter)
		{
			AddError(TEXT("the example's Torch Output Components are missing"));
			return true;
		}
		for (UTorchOutputComponent* C : { Classifier, Segmenter })
		{
			if (C->ResultCount > 0 && !C->LastResult.bSuccess)
			{
				AddError(FString::Printf(TEXT("%s: %s"), *C->GetName(), *C->LastResult.Error));
				return true;
			}
		}
		//Warm: both answering. Then measure for 3 seconds
		if (S->MeasureStart == 0.0)
		{
			if (Classifier->ResultCount >= 5 && Segmenter->ResultCount >= 5)
			{
				S->MeasureStart = Now;
				S->ClassifyAtStart = Classifier->ResultCount;
				S->SegmentAtStart = Segmenter->ResultCount;
			}
			else if (Now > S->Deadline)
			{
				AddError(FString::Printf(TEXT("no results: classifier %d, segmenter %d"), Classifier->ResultCount, Segmenter->ResultCount));
				return true;
			}
			return false;
		}
		if (!S->bScreenshot && Now - S->MeasureStart > 1.5)
		{
			S->bScreenshot = true;
			FScreenshotRequest::RequestScreenshot(TEXT("PyTorch_VisionExample.png"), false, false);
		}
		if (Now - S->MeasureStart < 3.0)
		{
			return false;
		}
		const double Seconds = Now - S->MeasureStart;
		const double ClassifyRate = (Classifier->ResultCount - S->ClassifyAtStart) / Seconds;
		const double SegmentRate = (Segmenter->ResultCount - S->SegmentAtStart) / Seconds;
		FString Label, Classes;
		UTorchBlueprintLibrary::GetOutputString(Classifier->LastResult, TEXT("text"), Label);
		UTorchBlueprintLibrary::GetOutputString(Segmenter->LastResult, TEXT("classes"), Classes);
		const FPythonStreamStats Cam = Classifier->ResolveStream() ? Classifier->ResolveStream()->GetStreamStats() : FPythonStreamStats();
		const FString Line = FString::Printf(TEXT("camera 512x512, captured every frame: classify %.1f/s (model %.2f ms), segment %.1f/s (model %.2f ms); camera stream: %lld sent, %lld dropped, ack latency %.1f ms, frame age at the model %.1f ms | ImageNet: %s | segments: %s"),
			ClassifyRate, Classifier->LastResult.ModelMs, SegmentRate, Segmenter->LastResult.ModelMs, Cam.Sent, Cam.Dropped, Cam.AckLatencyMs, Classifier->LastResult.FrameAgeMs, *Label, *Classes);
		AddInfo(Line);
		UE_LOG(LogTorch, Display, TEXT("BENCH vision %s"), *Line);
		TestTrue(FString::Printf(TEXT("real-time classification (%.1f/s)"), ClassifyRate), ClassifyRate >= 15.0);
		TestTrue(FString::Printf(TEXT("segmentation overlay at interactive rates (%.1f/s)"), SegmentRate), SegmentRate >= 10.0);
		TestFalse(TEXT("a label"), Label.IsEmpty());
		UTextureRenderTarget2D* Sink = Segmenter->RenderTargetSinks.FindRef(TEXT("image"));
		FPyImageOptions Options;
		const TArray<float> Px = UPythonImageLibrary::RenderTargetToTensor(Sink, Options).ToFloats();
		float Sum = 0.f;
		for (float V : Px)
		{
			Sum += V;
		}
		TestTrue(TEXT("the overlay was written into the render target"), Px.Num() > 0 && Sum / Px.Num() > 0.02f);
		return true;
	}));
	ADD_LATENT_AUTOMATION_COMMAND(FEndPlayMapCommand());
	return true;
}

#endif
