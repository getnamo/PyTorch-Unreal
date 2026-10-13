// Benchmarks (T0 spikes, kept as measurements): run with
//   UnrealEditor-Cmd.exe <Project>.uproject -ExecCmds="Automation RunTests PyTorch.Benchmarks; Quit" -unattended -nullrhi -nopause
// Results are logged as "BENCH ..." lines (LogTorch) and as test info.

#include "CoreMinimal.h"
#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS

#include "PyTorch.h"
#include "PythonComponent.h"
#include "PythonStreamComponent.h"
#include "PythonTypes.h"
#include "TorchTestHelpers.h"
#include "Dom/JsonObject.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "NNE.h"
#include "NNEModelData.h"
#include "NNERuntimeCPU.h"
#include "NNERuntimeGPU.h"
#include "NNETypes.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

using namespace TorchTests;

namespace TorchBenchmarks
{
	static void Report(FAutomationTestBase* Test, const FString& Line)
	{
		UE_LOG(LogTorch, Display, TEXT("BENCH %s"), *Line);
		Test->AddInfo(Line);
	}

	static FPyTensor RandomTensor(TArray<int64> Shape, int32 Seed)
	{
		FRandomStream Rng(Seed);
		int64 Num = 1;
		for (int64 D : Shape)
		{
			Num *= D;
		}
		TArray<float> Values;
		Values.SetNumUninitialized(Num);
		for (float& V : Values)
		{
			V = Rng.FRand();
		}
		return FPyTensor::FromFloats(Values, Shape);
	}
}

// ---------------------------------------------------------------------------------------------------------
// (a) N concurrent calls from Unreal into one model: one forward pass per call vs. micro-batches built from
// deferred replies.

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTorchBenchBatching, "PyTorch.Benchmarks.Batching", TorchTests::BenchmarkFlags)
bool FTorchBenchBatching::RunTest(const FString&)
{
	using namespace TorchBenchmarks;
	struct FState
	{
		FEnv Env;
		UPythonComponent* C = nullptr;
		TSharedPtr<FCall> Warmup;
		TSharedPtr<FCall> Stats;
		TArray<TSharedRef<FCall>> Calls;
		double Start = 0.0;
		TArray<double> Totals;
		TArray<double> Latencies;
	};
	const TSharedRef<FState> S = MakeShared<FState>();
	S->Env.Create();
	S->C = S->Env.Add<UPythonComponent>(S->Env.Spawn(), [](UPythonComponent* C) { C->ScriptParams.Script = TEXT("benchmarks/batching.py"); });
	Step([S] { S->Warmup = Call(S->C, TEXT("warmup"), TEXT("[]")); });
	WaitUntil(this, TEXT("warmup"), [S] { return S->Warmup->bDone; });
	Step([this, S]
	{
		TestTrue(TEXT("warmup: ") + S->Warmup->Result.Error, S->Warmup->Result.bSuccess);
		Report(this, TEXT("batching setup ") + S->Warmup->Result.JsonResult);
	});

	const FPyTensor Image = RandomTensor({ 28, 28 }, 1);
	const int32 Reps = 5;
	//classify_stacked: Unreal stacks the N requests of a tick into one (N, 28, 28) tensor and makes one call
	for (const TCHAR* Method : { TEXT("classify"), TEXT("classify_batched"), TEXT("classify_stacked") })
	{
		const bool bStacked = FCString::Strcmp(Method, TEXT("classify_stacked")) == 0;
		for (int32 N : { 1, 16, 64, 256 })
		{
			Step([S] { S->Totals.Reset(); S->Latencies.Reset(); Call(S->C, TEXT("reset_stats"), TEXT("[]")); });
			for (int32 Rep = 0; Rep < Reps; ++Rep)
			{
				Step([S, Method, N, Image, bStacked]
				{
					S->Calls.Reset();
					S->Start = FPlatformTime::Seconds();
					if (bStacked)
					{
						FPyTensor Stacked = Image;
						Stacked.Shape = { N, 28, 28 };
						Stacked.Data.Reset();
						for (int32 i = 0; i < N; ++i)
						{
							Stacked.Data.Append(Image.Data);
						}
						S->Calls.Add(Call(S->C, Method, TEXT("[]"), { Stacked }));
						return;
					}
					for (int32 i = 0; i < N; ++i)
					{
						S->Calls.Add(Call(S->C, Method, TEXT("[]"), { Image }));
					}
				});
				WaitUntil(this, FString::Printf(TEXT("%s x%d"), Method, N), [S]
				{
					for (const TSharedRef<FCall>& Ca : S->Calls)
					{
						if (!Ca->bDone)
						{
							return false;
						}
					}
					return true;
				}, 60.0);
				Step([this, S]
				{
					double Last = 0.0;
					for (const TSharedRef<FCall>& Ca : S->Calls)
					{
						if (!Ca->Result.bSuccess)
						{
							AddError(Ca->Result.Error);
							return;
						}
						S->Latencies.Add(Ca->Seconds * 1000.0);
						Last = FMath::Max(Last, Ca->Seconds);
					}
					S->Totals.Add(Last * 1000.0);
				});
			}
			Step([S] { S->Stats = Call(S->C, TEXT("get_stats"), TEXT("[]")); });
			WaitUntil(this, TEXT("stats"), [S] { return S->Stats->bDone; });
			Step([this, S, Method, N, Reps]
			{
				S->Totals.Sort();
				S->Latencies.Sort();
				const TSharedPtr<FJsonObject> Obj = S->Stats->Result.Result.IsValid() ? S->Stats->Result.Result->AsObject() : nullptr;
				const int32 Passes = Obj.IsValid() ? (int32)Obj->GetNumberField(TEXT("passes")) : -1;
				const int32 Rows = Obj.IsValid() ? (int32)Obj->GetNumberField(TEXT("rows")) : -1;
				TestEqual(FString::Printf(TEXT("%s x%d: every request answered by the model"), Method, N), Rows, N * Reps);
				const double MedianTotal = Percentile(S->Totals, 0.5);
				Report(this, FString::Printf(TEXT("batching method=%s n=%d all_done_ms(median)=%.2f calls_per_s=%.0f latency_ms p50=%.2f p99=%.2f forward_passes_per_burst=%.1f"),
					Method, N, MedianTotal, N / (MedianTotal / 1000.0), Percentile(S->Latencies, 0.5), Percentile(S->Latencies, 0.99), Passes / (double)Reps));
			});
		}
	}
	Step([S] { S->Env.Destroy(); });
	return true;
}

// ---------------------------------------------------------------------------------------------------------
// (d) A Lockstep stream as an environment step: all agents' observations in one frame, the policy's actions
// back. Policies: none (transport only), CPU, CUDA.

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTorchBenchLockstep, "PyTorch.Benchmarks.Lockstep", TorchTests::BenchmarkFlags)
bool FTorchBenchLockstep::RunTest(const FString&)
{
	using namespace TorchBenchmarks;
	struct FAgentEnv
	{
		FString Mode;
		UPythonComponent* C = nullptr;
		UPythonStreamComponent* Stream = nullptr;
		TSharedPtr<FCall> Info;
	};
	struct FState
	{
		FEnv Env;
		TArray<FAgentEnv> Envs;
	};
	constexpr int32 Obs = 64;
	constexpr int32 MaxAgents = 1024;
	const TSharedRef<FState> S = MakeShared<FState>();
	S->Env.Create();
	for (const TCHAR* Mode : { TEXT("none"), TEXT("cpu"), TEXT("cuda") })
	{
		FAgentEnv E;
		E.Mode = Mode;
		AActor* Actor = S->Env.Spawn();
		E.C = S->Env.Add<UPythonComponent>(Actor, [Mode](UPythonComponent* C)
		{
			C->ScriptParams.Script = TEXT("benchmarks/lockstep.py");
			C->ScriptParams.Args = { Mode };
		});
		UPythonComponent* Target = E.C;
		E.Stream = S->Env.Add<UPythonStreamComponent>(Actor, [Target](UPythonStreamComponent* St)
		{
			St->StreamName = TEXT("agents");
			St->Policy = EPythonStreamPolicy::Lockstep;
			St->Slots = 1;
			St->MaxFrameBytes = MaxAgents * Obs * sizeof(float);
			St->StepTimeoutSeconds = 30.f;
			St->PythonComponent = Target;
		});
		S->Envs.Add(E);
	}
	Step([S] { for (FAgentEnv& E : S->Envs) { E.Info = Call(E.C, TEXT("info"), TEXT("[]")); } });
	WaitUntil(this, TEXT("scripts"), [S] { for (FAgentEnv& E : S->Envs) { if (!E.Info->bDone) { return false; } } return true; });

	for (int32 EnvIndex = 0; EnvIndex < 3; ++EnvIndex)
	{
		for (int32 Agents : { 1, 64, 256, 1024 })
		{
			Step([this, S, EnvIndex, Agents]
			{
				FAgentEnv& E = S->Envs[EnvIndex];
				const FPyTensor Observations = RandomTensor({ Agents, Obs }, Agents);
				FPythonCallResult R;
				for (int32 i = 0; i < 20; ++i) //Warm up (first CUDA calls, allocator)
				{
					E.Stream->StepTensor(Observations, R);
				}
				TArray<double> Ms;
				const double End = FPlatformTime::Seconds() + 1.5;
				while (FPlatformTime::Seconds() < End)
				{
					const double T = FPlatformTime::Seconds();
					if (!E.Stream->StepTensor(Observations, R))
					{
						AddError(FString::Printf(TEXT("step failed (%s, %d agents): %s"), *E.Mode, Agents, *R.Error));
						return;
					}
					Ms.Add((FPlatformTime::Seconds() - T) * 1000.0);
				}
				const bool bShape = R.Tensors.Num() == 1 && R.Tensors[0].Shape == TArray<int64>({ Agents, 8 });
				TestTrue(FString::Printf(TEXT("actions shape (%s, %d agents)"), *E.Mode, Agents), bShape);
				Ms.Sort();
				double Sum = 0.0;
				for (double V : Ms)
				{
					Sum += V;
				}
				const double StepsPerSecond = Ms.Num() / (Sum / 1000.0);
				Report(this, FString::Printf(TEXT("lockstep policy=%s agents=%d steps_per_s=%.0f agent_steps_per_s=%.0f step_ms p50=%.3f p99=%.3f"),
					*E.Mode, Agents, StepsPerSecond, StepsPerSecond * Agents, Percentile(Ms, 0.5), Percentile(Ms, 0.99)));
			});
		}
	}
	Step([S] { S->Env.Destroy(); });
	return true;
}

// ---------------------------------------------------------------------------------------------------------
// (c) Export from torch (dynamo, opset 20, dynamic batch) and run in NNE: ORT CPU and ORT DirectML. Parity
// against torch's outputs, and latency.

namespace TorchBenchmarks
{
	struct FNNERun
	{
		TSharedPtr<UE::NNE::IModelInstanceRunSync> Instance;
		FString Error;
	};

	static FNNERun CreateInstance(UNNEModelData* Data, const FString& RuntimeName, bool bGpu)
	{
		FNNERun Out;
		if (bGpu)
		{
			TWeakInterfacePtr<INNERuntimeGPU> Runtime = UE::NNE::GetRuntime<INNERuntimeGPU>(RuntimeName);
			if (!Runtime.IsValid())
			{
				Out.Error = RuntimeName + TEXT(" (GPU) is not available");
				return Out;
			}
			if (Runtime->CanCreateModelGPU(Data) != INNERuntimeGPU::ECanCreateModelGPUStatus::Ok)
			{
				Out.Error = RuntimeName + TEXT(" can't create this model");
				return Out;
			}
			TSharedPtr<UE::NNE::IModelGPU> Model = Runtime->CreateModelGPU(Data);
			Out.Instance = Model.IsValid() ? Model->CreateModelInstanceGPU() : nullptr;
		}
		else
		{
			TWeakInterfacePtr<INNERuntimeCPU> Runtime = UE::NNE::GetRuntime<INNERuntimeCPU>(RuntimeName);
			if (!Runtime.IsValid())
			{
				Out.Error = RuntimeName + TEXT(" (CPU) is not available");
				return Out;
			}
			if (Runtime->CanCreateModelCPU(Data) != INNERuntimeCPU::ECanCreateModelCPUStatus::Ok)
			{
				Out.Error = RuntimeName + TEXT(" can't create this model");
				return Out;
			}
			TSharedPtr<UE::NNE::IModelCPU> Model = Runtime->CreateModelCPU(Data);
			Out.Instance = Model.IsValid() ? Model->CreateModelInstanceCPU() : nullptr;
		}
		if (!Out.Instance.IsValid())
		{
			Out.Error = RuntimeName + TEXT(": model instance creation failed");
		}
		return Out;
	}

	static TArray<uint32> ToShape(const TArray<TSharedPtr<FJsonValue>>& Values)
	{
		TArray<uint32> Shape;
		for (const TSharedPtr<FJsonValue>& V : Values)
		{
			Shape.Add((uint32)V->AsNumber());
		}
		return Shape;
	}
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTorchBenchNNE, "PyTorch.Benchmarks.NNE", TorchTests::BenchmarkFlags)
bool FTorchBenchNNE::RunTest(const FString&)
{
	using namespace TorchBenchmarks;
	struct FState
	{
		FEnv Env;
		UPythonComponent* C = nullptr;
		TSharedPtr<FCall> Export;
	};
	const TSharedRef<FState> S = MakeShared<FState>();
	const FString OutDir = FPaths::ConvertRelativePathToFull(FPaths::ProjectSavedDir() / TEXT("Torch/Benchmarks/onnx"));
	S->Env.Create();
	S->C = S->Env.Add<UPythonComponent>(S->Env.Spawn(), [](UPythonComponent* C) { C->ScriptParams.Script = TEXT("benchmarks/export_parity.py"); });
	Step([S, OutDir] { S->Export = Call(S->C, TEXT("export"), FString::Printf(TEXT("[\"%s\"]"), *OutDir.Replace(TEXT("\\"), TEXT("/")))); });
	WaitUntil(this, TEXT("export"), [S] { return S->Export->bDone; });
	Step([this, S]
	{
		if (!TestTrue(TEXT("export: ") + S->Export->Result.Error, S->Export->Result.bSuccess))
		{
			return;
		}
		for (const TSharedPtr<FJsonValue>& ModelValue : S->Export->Result.Result->AsArray())
		{
			const TSharedPtr<FJsonObject> M = ModelValue->AsObject();
			const FString Name = M->GetStringField(TEXT("name"));
			TArray64<uint8> Bytes;
			if (!TestTrue(TEXT("onnx file"), FFileHelper::LoadFileToArray(Bytes, *M->GetStringField(TEXT("onnx")))))
			{
				continue;
			}
			Report(this, FString::Printf(TEXT("nne model=%s onnx_bytes=%lld export_s=%.2f"), *Name, Bytes.Num(), M->GetNumberField(TEXT("export_s"))));
			UNNEModelData* Data = NewObject<UNNEModelData>();
			Data->Init(TEXT("onnx"), Bytes);
			for (const TPair<FString, bool>& R : TArray<TPair<FString, bool>>{ { TEXT("NNERuntimeORTCpu"), false }, { TEXT("NNERuntimeORTDml"), true } })
			{
				const double CreateStart = FPlatformTime::Seconds();
				FNNERun Run = CreateInstance(Data, R.Key, R.Value);
				const double CreateMs = (FPlatformTime::Seconds() - CreateStart) * 1000.0;
				if (!Run.Instance.IsValid())
				{
					AddWarning(FString::Printf(TEXT("%s: %s"), *Name, *Run.Error));
					continue;
				}
				for (const TSharedPtr<FJsonValue>& CaseValue : M->GetArrayField(TEXT("cases")))
				{
					const TSharedPtr<FJsonObject> Case = CaseValue->AsObject();
					const TArray<uint32> InShape = ToShape(Case->GetArrayField(TEXT("input_shape")));
					TArray<float> Input, Expected, Output;
					TArray<uint8> Raw;
					FFileHelper::LoadFileToArray(Raw, *Case->GetStringField(TEXT("input_file")));
					Input.SetNumUninitialized(Raw.Num() / sizeof(float));
					FMemory::Memcpy(Input.GetData(), Raw.GetData(), Raw.Num());
					FFileHelper::LoadFileToArray(Raw, *Case->GetStringField(TEXT("output_file")));
					Expected.SetNumUninitialized(Raw.Num() / sizeof(float));
					FMemory::Memcpy(Expected.GetData(), Raw.GetData(), Raw.Num());
					Output.SetNumZeroed(Expected.Num());

					const UE::NNE::FTensorShape Shape = UE::NNE::FTensorShape::Make(InShape);
					if (Run.Instance->SetInputTensorShapes({ Shape }) != UE::NNE::EResultStatus::Ok)
					{
						AddError(FString::Printf(TEXT("%s %s: SetInputTensorShapes failed"), *Name, *R.Key));
						continue;
					}
					const UE::NNE::FTensorBindingCPU In{ Input.GetData(), (uint64)Input.Num() * sizeof(float) };
					const UE::NNE::FTensorBindingCPU Out{ Output.GetData(), (uint64)Output.Num() * sizeof(float) };
					if (Run.Instance->RunSync({ In }, { Out }) != UE::NNE::EResultStatus::Ok)
					{
						AddError(FString::Printf(TEXT("%s %s: RunSync failed"), *Name, *R.Key));
						continue;
					}
					float MaxDiff = 0.f;
					float MaxAbs = 1.f;
					for (int32 i = 0; i < Output.Num(); ++i)
					{
						MaxDiff = FMath::Max(MaxDiff, FMath::Abs(Output[i] - Expected[i]));
						MaxAbs = FMath::Max(MaxAbs, FMath::Abs(Expected[i]));
					}
					TestTrue(FString::Printf(TEXT("%s %s batch %u matches torch (max diff %g)"), *Name, *R.Key, InShape[0], MaxDiff), MaxDiff < 1e-3f * MaxAbs);
					int32 Iters = 0;
					const double T = FPlatformTime::Seconds();
					while (Iters < 200 && (Iters < 5 || FPlatformTime::Seconds() - T < 1.0))
					{
						Run.Instance->RunSync({ In }, { Out });
						++Iters;
					}
					const double Ms = (FPlatformTime::Seconds() - T) * 1000.0 / Iters;
					Report(this, FString::Printf(TEXT("nne model=%s runtime=%s batch=%u ms=%.4f max_abs_diff=%g create_ms=%.1f torch_cpu_ms=%.4f torch_cuda_ms=%s"),
						*Name, *R.Key, InShape[0], Ms, MaxDiff, CreateMs, Case->GetNumberField(TEXT("torch_cpu_ms")),
						*PythonWire::ToJsonString(Case->TryGetField(TEXT("torch_cuda_ms")))));
				}
			}
		}
	});
	Step([S] { S->Env.Destroy(); });
	return true;
}

#endif
