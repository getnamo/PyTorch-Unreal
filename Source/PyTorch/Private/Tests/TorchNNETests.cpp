// PyTorch.NNE.*: Export To NNE (ut.export.onnx -> NNE model data) and Run Model on the NNE backend, against the
// same model on Python: examples/digits.py (an image model: Unreal converts the image like ut.Image) on ORT CPU and
// DirectML, one request and several stacked into a static batch. The reach policy on NNE: PyTorch.RL.PolicyOnNNE.

#include "CoreMinimal.h"
#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS && WITH_EDITOR

#include "PyTorch.h"
#include "TorchBlueprintLibrary.h"
#include "TorchModel.h"
#include "TorchNNE.h"
#include "TorchSubsystem.h"
#include "TorchTestHelpers.h"

using namespace TorchTests;

namespace TorchNNETests
{
	struct FRun
	{
		bool bDone = false;
		FTorchResult Result;
	};

	static TSharedRef<FRun> Run(UTorchModel* Model, const FString& Method, const TMap<FString, FPyTensor>& Inputs, const FString& Json = FString())
	{
		TSharedRef<FRun> Out = MakeShared<FRun>();
		UTorchSubsystem::Get()->Run(Model, Method, Inputs, Json, [Out](const FTorchResult& R) { Out->Result = R; Out->bDone = true; });
		return Out;
	}

	static bool AllDone(const TArray<TSharedRef<FRun>>& Runs)
	{
		for (const TSharedRef<FRun>& R : Runs)
		{
			if (!R->bDone)
			{
				return false;
			}
		}
		return true;
	}

	static float MaxDiff(const FTorchResult& A, const FTorchResult& B, const FString& Output)
	{
		const FPyTensor* TA = A.Outputs.Find(Output);
		const FPyTensor* TB = B.Outputs.Find(Output);
		if (!TA || !TB || TA->NumElements() != TB->NumElements())
		{
			return 1e9f;
		}
		const TArray<float> FA = TA->ToFloats();
		const TArray<float> FB = TB->ToFloats();
		float D = 0.f;
		for (int32 i = 0; i < FA.Num(); ++i)
		{
			D = FMath::Max(D, FMath::Abs(FA[i] - FB[i]));
		}
		return D;
	}
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTorchNNEDigitsTest, "PyTorch.NNE.Digits", TorchTests::Flags)
bool FTorchNNEDigitsTest::RunTest(const FString&)
{
	using namespace TorchNNETests;
	struct FState
	{
		UTorchModel* Model = nullptr;
		bool bExported = false;
		bool bExportOk = false;
		FString Summary;
		TArray<FPyTensor> Images;
		TArray<TSharedRef<FRun>> Samples, Python, Cpu, Stacked;
	};
	const TSharedRef<FState> S = MakeShared<FState>();
	S->Model = NewObject<UTorchModel>(GetTransientPackage(), TEXT("Test_NNEDigits"));
	S->Model->Script = TEXT("examples/digits.py");
	S->Model->ModelName = TEXT("digits");
	S->Model->ExportMethods = { TEXT("classify") };
	S->Model->ExportStaticBatch = 4;
	S->Model->AddToRoot();
	Step([S]
	{
		UTorchSubsystem::Get()->ExportToNNE(S->Model, [S](bool bOk, const FString& Summary) { S->bExportOk = bOk; S->Summary = Summary; S->bExported = true; });
		for (int32 i = 0; i < 6; ++i)
		{
			S->Samples.Add(Run(S->Model, TEXT("sample"), {}, FString::Printf(TEXT("{\"index\": %d, \"size\": 64}"), 100 + i)));
		}
	});
	WaitUntil(this, TEXT("export"), [S] { return S->bExported && AllDone(S->Samples); }, 600.0);
	Step([this, S]
	{
		AddInfo(TEXT("export: ") + S->Summary);
		TestTrue(TEXT("exported, and NNE matches torch on the reference rows: ") + S->Summary, S->bExportOk);
		const FTorchNNEMethod* M = S->Model->FindNNEMethod(TEXT("classify"));
		if (!TestNotNull(TEXT("classify's NNE export"), M))
		{
			return;
		}
		TestTrue(TEXT("a static batch-4 model too"), M->StaticModel != nullptr && M->StaticBatch == 4);
		TestTrue(FString::Printf(TEXT("parity on ORT CPU (%g)"), M->ParityMaxDiff), M->ParityMaxDiff >= 0.f && M->ParityMaxDiff < 1e-4f);
		for (const TSharedRef<FRun>& R : S->Samples)
		{
			FPyTensor Image;
			UTorchBlueprintLibrary::GetOutputTensor(R->Result, TEXT("image"), Image);
			S->Images.Add(Image);
		}
		//The same images through Python, then NNE (CPU), one request each
		S->Model->Backend = ETorchBackend::Python;
		for (const FPyTensor& Image : S->Images)
		{
			S->Python.Add(Run(S->Model, TEXT("classify"), { { TEXT("image"), Image } }));
		}
	});
	WaitUntil(this, TEXT("python"), [S] { return AllDone(S->Python); });
	Step([S]
	{
		S->Model->Backend = ETorchBackend::NNE;
		S->Model->NNERuntime = ETorchNNERuntime::CPU;
		for (const FPyTensor& Image : S->Images)
		{
			S->Cpu.Add(Run(S->Model, TEXT("classify"), { { TEXT("image"), Image } }));
		}
	});
	WaitUntil(this, TEXT("nne cpu"), [S] { return AllDone(S->Cpu); });
	Step([S]
	{
		S->Model->NNERuntime = ETorchNNERuntime::GPU;
		for (const FPyTensor& Image : S->Images) //Six requests in one tick: one stacked call, two static batches of 4
		{
			S->Stacked.Add(Run(S->Model, TEXT("classify"), { { TEXT("image"), Image } }));
		}
	});
	WaitUntil(this, TEXT("nne gpu"), [S] { return AllDone(S->Stacked); });
	Step([this, S]
	{
		for (int32 i = 0; i < S->Images.Num(); ++i)
		{
			const FTorchResult& P = S->Python[i]->Result;
			const FTorchResult& C = S->Cpu[i]->Result;
			const FTorchResult& G = S->Stacked[i]->Result;
			if (!TestTrue(TEXT("python: ") + P.Error, P.bSuccess) || !TestTrue(TEXT("nne cpu: ") + C.Error, C.bSuccess) || !TestTrue(TEXT("nne gpu: ") + G.Error, G.bSuccess))
			{
				return;
			}
			const int32 Digit = UTorchBlueprintLibrary::GetOutputInt(P, TEXT("digit"));
			const float DiffC = MaxDiff(P, C, TEXT("probabilities"));
			const float DiffG = MaxDiff(P, G, TEXT("probabilities"));
			AddInfo(FString::Printf(TEXT("image %d: python %d (%.3f); nne %s %d, %s %d; probability diff %.4f / %.4f; model %.3f ms / %.3f ms"), i, Digit,
				UTorchBlueprintLibrary::GetOutputFloat(P, TEXT("confidence")), *C.Backend, UTorchBlueprintLibrary::GetOutputInt(C, TEXT("digit")),
				*G.Backend, UTorchBlueprintLibrary::GetOutputInt(G, TEXT("digit")), DiffC, DiffG, C.ModelMs, G.ModelMs));
			TestEqual(TEXT("NNE CPU reads the same digit"), UTorchBlueprintLibrary::GetOutputInt(C, TEXT("digit")), Digit);
			TestEqual(TEXT("NNE GPU (stacked, static batch) reads the same digit"), UTorchBlueprintLibrary::GetOutputInt(G, TEXT("digit")), Digit);
			TestTrue(FString::Printf(TEXT("the image conversion matches ut.Image: probabilities within 0.02 (%.4f, %.4f)"), DiffC, DiffG), DiffC < 0.02f && DiffG < 0.02f);
		}
		TestTrue(TEXT("the backend is reported: ") + S->Cpu[0]->Result.Backend, S->Cpu[0]->Result.Backend == TEXT("nne:NNERuntimeORTCpu"));
		TestTrue(TEXT("six requests ran as one batch: ") + FString::FromInt(S->Stacked[0]->Result.BatchSize), S->Stacked[0]->Result.BatchSize == 6);
		TestTrue(TEXT("on the GPU runtime: ") + S->Stacked[0]->Result.Backend, S->Stacked[0]->Result.Backend == TEXT("nne:NNERuntimeORTDml"));
		S->Model->RemoveFromRoot();
	});
	return true;
}

#endif
