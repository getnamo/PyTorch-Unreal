// PyTorch.Model.*: Torch Model assets through UTorchSubsystem, against Content/Scripts/tests/torch_fixture.py
// and examples/digits.py. Engine-free coverage of unreal_torch: Content/Scripts/tests/torch_harness.py.

#include "CoreMinimal.h"
#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS

#include "PyTorch.h"
#include "PythonCLISettings.h"
#include "PythonEnvironment.h"
#include "TorchAsyncActions.h"
#include "TorchBlueprintLibrary.h"
#include "TorchModel.h"
#include "TorchSettings.h"
#include "TorchSubsystem.h"
#include "TorchTestHelpers.h"
#include "TorchTestReceiver.h"
#include "Dom/JsonObject.h"

using namespace TorchTests;

namespace TorchModelTests
{
	static UTorchModel* MakeModel(const FString& Script, const FString& Name)
	{
		UTorchModel* M = NewObject<UTorchModel>(GetTransientPackage(), *FString::Printf(TEXT("Test_%s"), *Name));
		M->Script = Script;
		M->ModelName = Name;
		M->AddToRoot();
		return M;
	}

	struct FRun
	{
		bool bDone = false;
		FTorchResult Result;
	};

	static TSharedRef<FRun> Run(UTorchModel* Model, const FString& Method, const TMap<FString, FPyTensor>& Inputs, const FString& Json = FString(), const FTorchRunOptions& Options = FTorchRunOptions())
	{
		TSharedRef<FRun> Out = MakeShared<FRun>();
		UTorchSubsystem::Get()->Run(Model, Method, Inputs, Json, [Out](const FTorchResult& R) { Out->Result = R; Out->bDone = true; }, Options);
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

	static const TCHAR* Fixture = TEXT("tests/torch_fixture.py");
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTorchModelStackingTest, "PyTorch.Model.Stacking", TorchTests::Flags)
bool FTorchModelStackingTest::RunTest(const FString&)
{
	using namespace TorchModelTests;
	UTorchModel* Doubler = MakeModel(Fixture, TEXT("doubler"));
	struct FState
	{
		TArray<TSharedRef<FRun>> Same, Other;
		TSharedPtr<FRun> Batched, Load;
	};
	const TSharedRef<FState> S = MakeShared<FState>();
	Step([S, Doubler]
	{
		S->Load = MakeShared<FRun>();
		UTorchSubsystem::Get()->LoadModel(Doubler, [S](bool bOk, const FString& Error) { S->Load->Result.bSuccess = bOk; S->Load->Result.Error = Error; S->Load->bDone = true; });
	});
	WaitUntil(this, TEXT("model loaded"), [S] { return S->Load->bDone; });
	Step([this, S, Doubler]
	{
		TestTrue(TEXT("loaded: ") + S->Load->Result.Error, S->Load->Result.bSuccess && UTorchSubsystem::Get()->IsModelReady(Doubler));
		AddInfo(TEXT("device: ") + UTorchSubsystem::Get()->GetModelDevice(Doubler));
		//One tick: three requests of shape [2] and a batched [4, 2] (same rows: one call of 7 rows), one of shape [3] (its own call)
		for (int32 i = 0; i < 3; ++i)
		{
			S->Same.Add(Run(Doubler, TEXT("double"), { { TEXT("x"), FPyTensor::FromFloats({ (float)i, 10.f + i }, { 2 }) } }));
		}
		S->Other.Add(Run(Doubler, TEXT("double"), { { TEXT("x"), FPyTensor::FromFloats({ 1.f, 2.f, 3.f }, { 3 }) } }));
		FTorchRunOptions Batched;
		Batched.bBatched = true;
		S->Batched = Run(Doubler, TEXT("double"), { { TEXT("x"), FPyTensor::FromFloats({ 1, 2, 3, 4, 5, 6, 7, 8 }, { 4, 2 }) } }, FString(), Batched);
	});
	WaitUntil(this, TEXT("results"), [S] { return AllDone(S->Same) && AllDone(S->Other) && S->Batched->bDone; });
	Step([this, S]
	{
		for (int32 i = 0; i < S->Same.Num(); ++i)
		{
			const FTorchResult& R = S->Same[i]->Result;
			const TArray<float> V = UTorchBlueprintLibrary::GetOutputFloats(R, TEXT(""));
			TestTrue(FString::Printf(TEXT("request %d: its own rows, no batch dimension: %s"), i, *R.Error), R.bSuccess && V == TArray<float>({ 2.f * i, 2.f * (10.f + i) })
				&& R.Outputs[TEXT("output")].Shape == TArray<int64>({ 2 }));
			TestEqual(TEXT("stacked with the tick's other requests of that row shape"), R.BatchSize, 7);
		}
		const FTorchResult& O = S->Other[0]->Result;
		TestTrue(TEXT("another shape: another call"), O.bSuccess && O.BatchSize == 1 && UTorchBlueprintLibrary::GetOutputFloats(O, TEXT("output")) == TArray<float>({ 2.f, 4.f, 6.f }));
		const FTorchResult& B = S->Batched->Result;
		TestTrue(TEXT("a batched request keeps its batch dimension"), B.bSuccess && B.Outputs[TEXT("output")].Shape == TArray<int64>({ 4, 2 }) && UTorchBlueprintLibrary::GetOutputFloat(B, TEXT("output"), 7) == 16.f);
	});
	Step([Doubler] { Doubler->RemoveFromRoot(); });
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTorchModelJsonTest, "PyTorch.Model.JsonAndRows", TorchTests::Flags)
bool FTorchModelJsonTest::RunTest(const FString&)
{
	using namespace TorchModelTests;
	AddExpectedMessage(TEXT("[python tests/torch_fixture.py]"), ELogVerbosity::Error, EAutomationExpectedMessageFlags::Contains, -1, false); //The failing calls
	UTorchModel* Rowwise = MakeModel(Fixture, TEXT("rowwise"));
	struct FState
	{
		TArray<TSharedRef<FRun>> Runs;
		TSharedPtr<FRun> BadMethod, BadInput, NoScript;
	};
	const TSharedRef<FState> S = MakeShared<FState>();
	UTorchModel* Empty = MakeModel(FString(), TEXT("none"));
	Step([S, Rowwise, Empty]
	{
		S->Runs.Add(Run(Rowwise, TEXT("stats"), { { TEXT("x"), FPyTensor::FromFloats({ 1.f, 2.f, 3.f }) } }, TEXT("{\"label\": \"a\"}")));
		S->Runs.Add(Run(Rowwise, TEXT("stats"), { { TEXT("x"), FPyTensor::FromFloats({ 4.f, 5.f, 6.f }) } }, TEXT("{\"label\": \"a\"}")));
		S->Runs.Add(Run(Rowwise, TEXT("stats"), { { TEXT("x"), FPyTensor::FromFloats({ 7.f, 8.f, 9.f }) } }, TEXT("{\"label\": \"b\"}")));
		S->BadMethod = Run(Rowwise, TEXT("nope"), {});
		S->BadInput = Run(Rowwise, TEXT("stats"), { { TEXT("y"), FPyTensor::FromFloats({ 1.f }) } });
		S->NoScript = Run(Empty, TEXT("x"), {});
	});
	WaitUntil(this, TEXT("results"), [S] { return AllDone(S->Runs) && S->BadMethod->bDone && S->BadInput->bDone && S->NoScript->bDone; });
	Step([this, S, Rowwise, Empty]
	{
		const float Sums[] = { 6.f, 15.f, 24.f };
		const TCHAR* Labels[] = { TEXT("a"), TEXT("a"), TEXT("b") };
		for (int32 i = 0; i < 3; ++i)
		{
			const FTorchResult& R = S->Runs[i]->Result;
			FString Label;
			TestTrue(FString::Printf(TEXT("row %d: per-request tensor and JSON outputs: %s"), i, *R.Error), R.bSuccess
				&& UTorchBlueprintLibrary::GetOutputFloat(R, TEXT("sum")) == Sums[i] && UTorchBlueprintLibrary::GetOutputInt(R, TEXT("n")) == 3
				&& UTorchBlueprintLibrary::GetOutputString(R, TEXT("label"), Label) && Label == Labels[i]);
		}
		TestEqual(TEXT("same JSON inputs stack; other JSON inputs go separately"), S->Runs[0]->Result.BatchSize, 2);
		TestTrue(TEXT("unknown method fails"), !S->BadMethod->Result.bSuccess && S->BadMethod->Result.Error.Contains(TEXT("no exported method 'nope'")));
		TestTrue(TEXT("unknown input fails"), !S->BadInput->Result.bSuccess && S->BadInput->Result.Error.Contains(TEXT("has no input y")));
		TestTrue(TEXT("a model without a script fails at once"), !S->NoScript->Result.bSuccess);
		Rowwise->RemoveFromRoot();
		Empty->RemoveFromRoot();
	});
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTorchModelConcurrentTest, "PyTorch.Model.ConcurrentCallers", TorchTests::Flags)
bool FTorchModelConcurrentTest::RunTest(const FString&)
{
	using namespace TorchModelTests;
	//T2's exit check: a CUDA model serves 64 concurrent Blueprint callers (Run Model nodes) in batched passes
	UTorchModel* Digits = MakeModel(TEXT("examples/digits.py"), TEXT("digits"));
	struct FState
	{
		TSharedPtr<FRun> Load;
		UTorchTestReceiver* Receiver = nullptr;
		TSharedPtr<FPythonCallResult> Stats;
	};
	const TSharedRef<FState> S = MakeShared<FState>();
	S->Receiver = NewObject<UTorchTestReceiver>();
	S->Receiver->AddToRoot();
	Step([S, Digits]
	{
		S->Load = MakeShared<FRun>();
		UTorchSubsystem::Get()->LoadModel(Digits, [S](bool bOk, const FString& Error) { S->Load->Result.bSuccess = bOk; S->Load->Result.Error = Error; S->Load->bDone = true; });
	});
	WaitUntil(this, TEXT("digits loaded"), [S] { return S->Load->bDone; });
	for (int32 Round = 0; Round < 3; ++Round)
	{
		Step([S, Digits, Round]
		{
			UTorchSubsystem::Get()->GetStats(Digits, [](const FPythonCallResult&) {}, /*bReset*/ true);
			S->Receiver->Results.Reset();
			FRandomStream Rng(Round);
			for (int32 i = 0; i < 64; ++i)
			{
				TArray<float> Pixels;
				Pixels.SetNumUninitialized(28 * 28);
				for (float& P : Pixels)
				{
					P = Rng.FRand();
				}
				TMap<FString, FPyTensor> Inputs;
				Inputs.Add(TEXT("image"), FPyTensor::FromFloats(Pixels, { 28, 28, 1 }));
				//What a Blueprint's Run Model node does: its On Success / On Failure pins get the result
				UTorchRunModelAsyncAction* Node = UTorchRunModelAsyncAction::RunModel(Digits, TEXT("classify"), Inputs, FString());
				Node->OnSuccess.AddDynamic(S->Receiver, &UTorchTestReceiver::OnResult);
				Node->OnFailure.AddDynamic(S->Receiver, &UTorchTestReceiver::OnResult);
				Node->Activate();
			}
		});
		WaitUntil(this, TEXT("64 callers"), [S] { return S->Receiver->Results.Num() == 64; }, 60.0);
		Step([S, Digits]
		{
			S->Stats = MakeShared<FPythonCallResult>();
			UTorchSubsystem::Get()->GetStats(Digits, [Stats = S->Stats](const FPythonCallResult& R) { *Stats = R; });
		});
		WaitUntil(this, TEXT("stats"), [S] { return S->Stats->bSuccess || !S->Stats->Error.IsEmpty(); });
		Step([this, S, Digits, Round]
		{
			TArray<double> Ms;
			int32 Ok = 0, MaxBatch = 0;
			const TArray<FTorchResult>& Results = S->Receiver->Results;
			for (const FTorchResult& R : Results)
			{
				Ok += R.bSuccess ? 1 : 0;
				Ms.Add(R.LatencyMs);
				MaxBatch = FMath::Max(MaxBatch, R.BatchSize);
				if (!R.bSuccess)
				{
					AddError(R.Error);
					break;
				}
			}
			Ms.Sort();
			const TSharedPtr<FJsonObject> Stats = S->Stats->Result.IsValid() ? S->Stats->Result->AsObject() : nullptr;
			const int32 Passes = Stats.IsValid() ? (int32)Stats->GetNumberField(TEXT("passes")) : -1;
			TestEqual(TEXT("every caller answered"), Ok, 64);
			TestTrue(TEXT("each answer is a digit"), Results.Num() == 64 && Results[0].Outputs.Contains(TEXT("digit")) && Results[0].Outputs[TEXT("probabilities")].Shape == TArray<int64>({ 10 }));
			TestTrue(FString::Printf(TEXT("batched: %d forward pass(es) for 64 callers"), Passes), Passes >= 1 && Passes <= 2 && MaxBatch == 64);
			const FString Line = FString::Printf(TEXT("digits on %s: 64 Run Model callers in one tick -> %d forward pass(es), batch %d, latency p50 %.1f ms, max %.1f ms, model %.2f ms"),
				*UTorchSubsystem::Get()->GetModelDevice(Digits), Passes, MaxBatch, Percentile(Ms, 0.5), Ms.Num() ? Ms.Last() : 0.0, Results.Num() ? Results[0].ModelMs : 0.f);
			AddInfo(Line);
			UE_LOG(LogTorch, Display, TEXT("BENCH round %d %s"), Round, *Line);
		});
	}
	Step([S, Digits]
	{
		Digits->RemoveFromRoot();
		S->Receiver->RemoveFromRoot();
	});
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTorchModelSignatureTest, "PyTorch.Model.Signature", TorchTests::Flags)
bool FTorchModelSignatureTest::RunTest(const FString&)
{
	using namespace TorchModelTests;
	UTorchModel* Rowwise = MakeModel(Fixture, TEXT("rowwise"));
	struct FState
	{
		bool bDone = false;
		bool bOk = false;
		FString Error;
	};
	const TSharedRef<FState> S = MakeShared<FState>();
	Step([S, Rowwise]
	{
		UTorchSubsystem::Get()->FetchSignature(Rowwise, [S](bool bOk, const FString& Error) { S->bOk = bOk; S->Error = Error; S->bDone = true; });
	});
	WaitUntil(this, TEXT("signature"), [S] { return S->bDone; });
	Step([this, S, Rowwise]
	{
		TestTrue(TEXT("signature fetched: ") + S->Error, S->bOk);
		const FTorchMethodSpec* Stats = Rowwise->FindMethod(TEXT("stats"));
		TestTrue(TEXT("method and typed inputs"), Stats && !Stats->bBatched && Stats->Inputs.Num() == 2 && Stats->Inputs[0].Kind == TEXT("tensor")
			&& Stats->Inputs[0].Type == TEXT("float32") && Stats->Inputs[0].Shape == TArray<int64>({ -1 }) && Stats->Inputs[1].Kind == TEXT("json") && !Stats->Inputs[1].bRequired);
		Rowwise->RemoveFromRoot();
	});
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTorchModelReloadTest, "PyTorch.Model.ReloadKeepsWeights", TorchTests::Flags)
bool FTorchModelReloadTest::RunTest(const FString&)
{
	using namespace TorchModelTests;
	UTorchModel* Counter = MakeModel(Fixture, TEXT("counter"));
	struct FState
	{
		TSharedPtr<FRun> First, Second;
	};
	const TSharedRef<FState> S = MakeShared<FState>();
	Step([S, Counter] { S->First = Run(Counter, TEXT("bump"), {}, TEXT("{\"amount\": 3}")); });
	WaitUntil(this, TEXT("bump"), [S] { return S->First->bDone; });
	Step([Counter] { UTorchSubsystem::Get()->ReloadScript(Counter); });
	WaitUntil(this, TEXT("reloaded"), [Counter] { return UTorchSubsystem::Get()->GetService(Counter)->IsRunning(); });
	Step([S, Counter] { S->Second = Run(Counter, TEXT("bump"), {}, TEXT("{\"amount\": 1}")); });
	WaitUntil(this, TEXT("bump after reload"), [S] { return S->Second->bDone; });
	Step([this, S, Counter]
	{
		const float A = UTorchBlueprintLibrary::GetOutputFloat(S->First->Result, TEXT("output"));
		const float B = UTorchBlueprintLibrary::GetOutputFloat(S->Second->Result, TEXT("output"));
		TestTrue(FString::Printf(TEXT("weights kept across a script reload (%f -> %f)"), A, B), S->Second->Result.bSuccess && FMath::IsNearlyEqual(B, A + 1.f, 1e-4f));
		Counter->RemoveFromRoot();
	});
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTorchSettingsTest, "PyTorch.Settings.Profile", TorchTests::Flags)
bool FTorchSettingsTest::RunTest(const FString&)
{
	const UTorchSettings* Settings = GetDefault<UTorchSettings>();
	TestEqual(TEXT("profile backend"), UTorchSettings::TorchBackend(ETorchProfile::CUDA130), FString(TEXT("cu130")));
	if (Settings->Profile != ETorchProfile::None)
	{
		TestFalse(TEXT("a torch backend is set for uv"), GetDefault<UPythonCLISettings>()->TorchBackend.IsEmpty());
	}
	const TMap<FString, FString> Env = FPythonEnvironment::GetEnvironmentVariables();
	TestTrue(TEXT("compile caches in the project"), Env.Contains(TEXT("TORCHINDUCTOR_CACHE_DIR")) && Env.Contains(TEXT("TRITON_CACHE_DIR")));
	return true;
}

#endif
