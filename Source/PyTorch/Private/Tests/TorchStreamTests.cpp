// PyTorch.Streams.*: Torch Output Components running model methods on Python stream components (no scripts):
// pushed audio into examples/audio.py, a render target into the fixture's image models (with a render-target
// sink). The render-target test needs a real RHI (-RenderOffscreen).

#include "CoreMinimal.h"
#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS

#include "PyTorch.h"
#include "PythonAudioStreamComponent.h"
#include "PythonImageLibrary.h"
#include "PythonRenderTargetStreamComponent.h"
#include "TorchBlueprintLibrary.h"
#include "TorchModel.h"
#include "TorchOutputComponent.h"
#include "TorchSubsystem.h"
#include "TorchTestHelpers.h"
#include "DynamicRHI.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Misc/App.h"
#include "RenderingThread.h"

using namespace TorchTests;

namespace TorchStreamTests
{
	static UTorchModel* MakeModel(const FString& Script, const FString& Name)
	{
		UTorchModel* M = NewObject<UTorchModel>(GetTransientPackage(), *FString::Printf(TEXT("Test_Stream_%s"), *Name));
		M->Script = Script;
		M->ModelName = Name;
		M->AddToRoot();
		return M;
	}

	/** An actor with a stream component and a Torch Output Component, registered together before BeginPlay (as
	 when they're on a Blueprint). */
	template<typename TStream>
	static TPair<TStream*, UTorchOutputComponent*> MakeRig(FEnv& Env, TFunctionRef<void(TStream*)> ConfigureStream, TFunctionRef<void(UTorchOutputComponent*)> ConfigureOutput)
	{
		AActor* Actor = Env.Spawn();
		TStream* Stream = NewObject<TStream>(Actor, TEXT("Stream"));
		ConfigureStream(Stream);
		UTorchOutputComponent* Out = NewObject<UTorchOutputComponent>(Actor, TEXT("Output"));
		ConfigureOutput(Out);
		Out->StreamComponent = TEXT("Stream");
		Stream->RegisterComponent();
		Out->RegisterComponent();
		Actor->DispatchBeginPlay();
		return { Stream, Out };
	}
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTorchStreamAudioTest, "PyTorch.Streams.AudioToModel", TorchTests::Flags)
bool FTorchStreamAudioTest::RunTest(const FString&)
{
	using namespace TorchStreamTests;
	struct FState
	{
		FEnv Env;
		UPythonAudioStreamComponent* Audio = nullptr;
		UTorchOutputComponent* Out = nullptr;
		UTorchModel* Model = nullptr;
		bool bLoaded = false;
		int32 Pushed = 0;
	};
	const TSharedRef<FState> S = MakeShared<FState>();
	S->Env.Create();
	S->Model = MakeModel(TEXT("examples/audio.py"), TEXT("audio_features"));
	UTorchModel* Model = S->Model;
	const TPair<UPythonAudioStreamComponent*, UTorchOutputComponent*> Rig = MakeRig<UPythonAudioStreamComponent>(S->Env,
		[](UPythonAudioStreamComponent* A)
		{
			A->StreamName = TEXT("mic");
			A->Source = EPythonAudioSource::Manual;
			A->ChunkSeconds = 0.1f;
			A->bDownmixToMono = true;
			A->Policy = EPythonStreamPolicy::Queue;
		},
		[Model](UTorchOutputComponent* O)
		{
			O->Model = Model;
			O->Method = TEXT("analyze");
			O->InputName = TEXT("audio");
		});
	S->Audio = Rig.Key;
	S->Out = Rig.Value;
	Step([S] { UTorchSubsystem::Get()->LoadModel(S->Model, [S](bool, const FString&) { S->bLoaded = true; }); });
	WaitUntil(this, TEXT("model"), [S] { return S->bLoaded; });
	//One second of a 440 Hz sine at half amplitude (rms -9 dBFS), in 0.1 s chunks
	WaitUntil(this, TEXT("results"), [S]
	{
		if (S->Pushed < 10)
		{
			constexpr int32 Rate = 48000;
			TArray<float> Chunk;
			Chunk.SetNumUninitialized(Rate / 10);
			for (int32 i = 0; i < Chunk.Num(); ++i)
			{
				Chunk[i] = 0.5f * FMath::Sin(2.f * PI * 440.f * (S->Pushed * Chunk.Num() + i) / Rate);
			}
			S->Audio->PushAudio(Chunk, 1, Rate);
			++S->Pushed;
		}
		return S->Out->ResultCount >= 8;
	}, 60.0);
	Step([this, S]
	{
		const FTorchResult& R = S->Out->LastResult;
		const float Pitch = UTorchBlueprintLibrary::GetOutputFloat(R, TEXT("pitch"));
		const float Loudness = UTorchBlueprintLibrary::GetOutputFloat(R, TEXT("loudness"));
		FString Text;
		UTorchBlueprintLibrary::GetOutputString(R, TEXT("text"), Text);
		TestTrue(FString::Printf(TEXT("pitch of the pushed sine: %.1f Hz (%s)"), Pitch, *R.Error), R.bSuccess && FMath::Abs(Pitch - 440.f) < 5.f);
		TestTrue(FString::Printf(TEXT("loudness: %.1f dB"), Loudness), FMath::Abs(Loudness + 9.03f) < 1.f);
		AddInfo(FString::Printf(TEXT("%d results; last: %s, model %.2f ms"), S->Out->ResultCount, *Text, R.ModelMs));
		S->Env.Destroy();
		S->Model->RemoveFromRoot();
	});
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTorchStreamRenderTargetTest, "PyTorch.Streams.RenderTargetToModel", TorchTests::Flags)
bool FTorchStreamRenderTargetTest::RunTest(const FString&)
{
	using namespace TorchStreamTests;
	if (!FApp::CanEverRender() || !GDynamicRHI || FCString::Strcmp(GDynamicRHI->GetName(), TEXT("Null")) == 0)
	{
		AddWarning(TEXT("needs a real RHI (run without -nullrhi, e.g. -RenderOffscreen): skipped"));
		return true;
	}
	struct FState
	{
		FEnv Env;
		UPythonRenderTargetStreamComponent* Cam = nullptr;
		UTorchOutputComponent* Out = nullptr;
		UTorchModel* Model = nullptr;
		UTextureRenderTarget2D* Source = nullptr;
		UTextureRenderTarget2D* Sink = nullptr;
		bool bLoaded = false;
		int32 Captures = 0;
	};
	const TSharedRef<FState> S = MakeShared<FState>();
	S->Env.Create();
	S->Model = MakeModel(TEXT("tests/torch_fixture.py"), TEXT("imager"));
	auto MakeRT = [](int32 Size, FLinearColor Clear)
	{
		UTextureRenderTarget2D* RT = NewObject<UTextureRenderTarget2D>();
		RT->RenderTargetFormat = RTF_RGBA8;
		RT->bForceLinearGamma = true;
		RT->ClearColor = Clear;
		RT->InitAutoFormat(Size, Size);
		RT->UpdateResourceImmediate(true);
		RT->AddToRoot();
		return RT;
	};
	S->Source = MakeRT(64, FLinearColor(1.f, 0.5f, 0.f, 1.f));
	S->Sink = MakeRT(16, FLinearColor::Black);
	UTorchModel* Model = S->Model;
	UTextureRenderTarget2D* Source = S->Source;
	UTextureRenderTarget2D* Sink = S->Sink;
	const TPair<UPythonRenderTargetStreamComponent*, UTorchOutputComponent*> Rig = MakeRig<UPythonRenderTargetStreamComponent>(S->Env,
		[Source](UPythonRenderTargetStreamComponent* C)
		{
			C->StreamName = TEXT("camera");
			C->RenderTarget = Source;
			C->bAutoCapture = false;
			C->Policy = EPythonStreamPolicy::Latest;
		},
		[Model, Sink](UTorchOutputComponent* O)
		{
			O->Model = Model;
			O->Method = TEXT("invert");
			O->InputName = TEXT("image");
			O->RenderTargetSinks.Add(TEXT("image"), Sink);
		});
	S->Cam = Rig.Key;
	S->Out = Rig.Value;
	Step([S] { UTorchSubsystem::Get()->LoadModel(S->Model, [S](bool, const FString&) { S->bLoaded = true; }); });
	WaitUntil(this, TEXT("model"), [S] { return S->bLoaded; });
	WaitUntil(this, TEXT("results"), [S]
	{
		if (S->Captures < 20 && S->Cam->Capture())
		{
			++S->Captures;
		}
		S->Cam->PollReadbacks();
		return S->Out->ResultCount >= 3;
	}, 60.0);
	Step([this, S]
	{
		FlushRenderingCommands();
		const FTorchResult& R = S->Out->LastResult;
		const float Mean = UTorchBlueprintLibrary::GetOutputFloat(R, TEXT("mean"));
		TestTrue(FString::Printf(TEXT("the model saw the render target: mean %.3f (%s)"), Mean, *R.Error), R.bSuccess && FMath::Abs(Mean - 0.5f) < 0.01f);
		TestTrue(TEXT("the sink resized to the output image"), S->Sink->SizeX == 64 && S->Sink->SizeY == 64);
		FPyImageOptions Options;
		const TArray<float> Px = UPythonImageLibrary::RenderTargetToTensor(S->Sink, Options).ToFloats();
		TestTrue(TEXT("the sink holds the inverted image (0, ~0.5, 1)"), Px.Num() >= 3 && Px[0] < 0.01f && FMath::Abs(Px[1] - 0.5f) < 0.02f && Px[2] > 0.99f);
		AddInfo(FString::Printf(TEXT("%d results from %d captures (Latest skips frames while one is in flight); model %.2f ms"), S->Out->ResultCount, S->Captures, R.ModelMs));
		S->Env.Destroy();
		S->Model->RemoveFromRoot();
		S->Source->RemoveFromRoot();
		S->Sink->RemoveFromRoot();
	});
	return true;
}

#endif
