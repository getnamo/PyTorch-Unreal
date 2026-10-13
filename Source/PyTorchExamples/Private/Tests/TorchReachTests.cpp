// PyTorch.RL.*: reinforcement learning on the reach task (ATorchReachEnvironment, examples/reach.py): the
// environment without Python (heuristic), recording and behavior cloning, PPO training to a target return, and
// deterministic replay of the trained policy with a fixed seed. The test world doesn't tick: the tests step the
// environment themselves.

#include "CoreMinimal.h"
#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS

#include "PythonJob.h"
#include "TorchModel.h"
#include "TorchReachExample.h"
#include "TorchSubsystem.h"
#include "TorchTestHelpers.h"
#include "Dom/JsonObject.h"
#include "HAL/FileManager.h"
#include "Misc/Paths.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

using namespace TorchTests;

namespace TorchReachTests
{
	static UTorchModel* ReachPolicy()
	{
		static TWeakObjectPtr<UTorchModel> Model;
		if (!Model.IsValid())
		{
			UTorchModel* M = NewObject<UTorchModel>(GetTransientPackage(), TEXT("Test_ReachPolicy"));
			M->Script = TEXT("examples/reach.py");
			M->ModelName = TEXT("reach_policy");
			M->AddToRoot();
			Model = M;
		}
		return Model.Get();
	}

	/** An environment configured before its BeginPlay (which spawns the agents). */
	static ATorchReachEnvironment* SpawnEnv(FEnv& Env, TFunctionRef<void(ATorchReachEnvironment*)> Configure)
	{
		ATorchReachEnvironment* E = Env.World->SpawnActorDeferred<ATorchReachEnvironment>(ATorchReachEnvironment::StaticClass(), FTransform::Identity);
		Configure(E);
		E->FinishSpawning(FTransform::Identity);
		E->DispatchBeginPlay();
		Env.Actors.Add(E);
		return E;
	}

	/** Steps the environment from latent commands (up to StepsPerFrame a frame) until Condition. */
	static void Drive(FAutomationTestBase* Test, const FString& Label, ATorchEnvironment* E, int32 StepsPerFrame, TFunction<bool()> Condition, double InTimeout)
	{
		WaitUntil(Test, Label, [E, StepsPerFrame, Condition]
		{
			for (int32 i = 0; i < StepsPerFrame && !Condition(); ++i)
			{
				if (!E->StepEnvironment())
				{
					break;
				}
			}
			return Condition();
		}, InTimeout);
	}

	static int64 Steps(const ATorchEnvironment* E)
	{
		int64 S;
		int32 Episodes;
		float Mean;
		E->GetStats(S, Episodes, Mean);
		return S;
	}

	static int32 Episodes(const ATorchEnvironment* E)
	{
		int64 S;
		int32 Count;
		float Mean;
		E->GetStats(S, Count, Mean);
		return Count;
	}

	static float MeanReturn(const ATorchEnvironment* E)
	{
		int64 S;
		int32 Count;
		float Mean;
		E->GetStats(S, Count, Mean);
		return Mean;
	}

	static double ResultNumber(const UPythonJob* Job, const FString& Field)
	{
		TSharedPtr<FJsonObject> Obj;
		double Value = -1.0;
		if (FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Job->ResultJson), Obj) && Obj.IsValid())
		{
			Obj->TryGetNumberField(Field, Value);
		}
		return Value;
	}

	/** Every agent's position after each step: a run's trajectory. */
	struct FTrajectory
	{
		TArray<FVector2D> Positions;
		int64 LastStep = -1;

		void Sample(ATorchEnvironment* E)
		{
			const int64 S = Steps(E);
			if (S == LastStep)
			{
				return;
			}
			LastStep = S;
			for (UTorchAgentComponent* A : E->GetAgents())
			{
				Positions.Add(CastChecked<UTorchReachAgentComponent>(A)->Position);
			}
		}
	};
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTorchReachHeuristicTest, "PyTorch.RL.Heuristic", TorchTests::Flags)
bool FTorchReachHeuristicTest::RunTest(const FString&)
{
	using namespace TorchReachTests;
	FEnv Env;
	Env.Create();
	ATorchReachEnvironment* E = SpawnEnv(Env, [](ATorchReachEnvironment* R) { R->NumAgents = 8; R->Mode = ETorchEnvironmentMode::Heuristic; R->Seed = 3; });
	TestEqual(TEXT("agents spawned and registered"), E->GetAgents().Num(), 8);
	TestEqual(TEXT("observation size from the spec"), E->GetObservationSize(), 4);
	for (int32 i = 0; i < 400; ++i)
	{
		TestTrue(TEXT("a heuristic step"), E->StepEnvironment());
	}
	const float Mean = MeanReturn(E);
	AddInfo(FString::Printf(TEXT("scripted expert: %d episodes, mean return %.2f"), Episodes(E), Mean));
	TestTrue(TEXT("episodes end (reached) and restart in the same step"), Episodes(E) > 50);
	TestTrue(FString::Printf(TEXT("the scripted expert's mean return is high (%.2f)"), Mean), Mean > 1.5f);
	//Off: nothing moves
	E->SetMode(ETorchEnvironmentMode::Off);
	TestFalse(TEXT("Off doesn't step"), E->StepEnvironment());
	Env.Destroy();
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTorchReachRecordTest, "PyTorch.RL.RecordAndClone", TorchTests::Flags)
bool FTorchReachRecordTest::RunTest(const FString&)
{
	using namespace TorchReachTests;
	struct FState
	{
		FEnv Env;
		ATorchReachEnvironment* E = nullptr;
		FString Dir;
		UPythonJob* Job = nullptr;
		int32 RunEpisodesFrom = 0;
	};
	const TSharedRef<FState> S = MakeShared<FState>();
	S->Env.Create();
	S->E = SpawnEnv(S->Env, [](ATorchReachEnvironment* R)
	{
		R->NumAgents = 32;
		R->Mode = ETorchEnvironmentMode::Record;
		R->RecordingName = TEXT("ReachTest");
		R->StepsPerShard = 100;
		R->PolicyModel = ReachPolicy();
	});
	for (int32 i = 0; i < 251; ++i)
	{
		S->E->StepEnvironment();
	}
	S->E->SetMode(ETorchEnvironmentMode::Off); //Flushes the last shard
	TArray<FString> Sessions;
	const FString Root = FPaths::ProjectSavedDir() / TEXT("Torch/Recordings/ReachTest");
	IFileManager::Get().FindFiles(Sessions, *(Root / TEXT("*")), false, true);
	Sessions.Sort();
	TArray<FString> Shards;
	if (Sessions.Num())
	{
		IFileManager::Get().FindFiles(Shards, *(Root / Sessions.Last() / TEXT("shard_*.safetensors")), true, false);
	}
	TestEqual(TEXT("250 transitions in shards of 100 steps: 3 shards"), Shards.Num(), 3);
	Step([S]
	{
		S->Job = UTorchSubsystem::Get()->StartJob(ReachPolicy(), TEXT("train_reach_bc"), TEXT("{\"recording\": \"ReachTest\", \"sessions\": \"latest\", \"epochs\": 15}"), false, true);
		S->Job->AddToRoot();
	});
	WaitUntil(this, TEXT("behavior cloning"), [S] { return S->Job->IsFinished(); }, 600.0);
	Step([this, S]
	{
		TestEqual(TEXT("train_reach_bc completed: ") + S->Job->Error, S->Job->State, EPythonJobState::Completed);
		AddInfo(TEXT("train_reach_bc: ") + S->Job->ResultJson);
		TestEqual(TEXT("8000 recorded transitions (32 agents x 250 steps)"), ResultNumber(S->Job, TEXT("samples")), 8000.0);
		TestTrue(TEXT("the actions are fitted"), ResultNumber(S->Job, TEXT("loss")) >= 0.0 && ResultNumber(S->Job, TEXT("loss")) < 0.05);
		S->E->SetMode(ETorchEnvironmentMode::Run);
		S->RunEpisodesFrom = Episodes(S->E);
	});
	Drive(this, TEXT("the cloned policy running"), S->E, 4, [S] { return Episodes(S->E) >= S->RunEpisodesFrom + 150; }, 300.0);
	Step([this, S]
	{
		const float Mean = MeanReturn(S->E);
		AddInfo(FString::Printf(TEXT("cloned policy: mean return %.2f over the last 100 episodes"), Mean));
		TestTrue(FString::Printf(TEXT("the cloned policy reaches its targets (mean return %.2f)"), Mean), Mean > 1.0f);
		S->Job->RemoveFromRoot();
		S->Env.Destroy();
	});
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTorchReachTrainTest, "PyTorch.RL.TrainAndReplay", TorchTests::Flags)
bool FTorchReachTrainTest::RunTest(const FString&)
{
	using namespace TorchReachTests;
	struct FState
	{
		FEnv Env;
		ATorchReachEnvironment* E = nullptr;
		double Started = 0.0;
		int32 RunEpisodesFrom = 0;
		FTrajectory A, B, C;
		int64 From = 0;
	};
	const TSharedRef<FState> S = MakeShared<FState>();
	S->Env.Create();
	S->E = SpawnEnv(S->Env, [](ATorchReachEnvironment* R)
	{
		R->NumAgents = 64;
		R->Mode = ETorchEnvironmentMode::Train;
		R->TrainingModel = ReachPolicy();
		R->PolicyModel = ReachPolicy();
		R->TrainingJob = TEXT("train_reach");
		R->TrainingParamsJson = TEXT("{\"total_steps\": 1500000, \"target_return\": 1.2}");
	});
	S->Started = FPlatformTime::Seconds();
	TestNotNull(TEXT("the training job started on BeginPlay"), S->E->GetTrainingJob());
	if (S->E->GetTrainingJob())
	{
		S->E->GetTrainingJob()->AddToRoot();
	}
	Drive(this, TEXT("PPO training"), S->E, 256, [S] { return S->E->Mode != ETorchEnvironmentMode::Train; }, 900.0);
	WaitUntil(this, TEXT("the job's end"), [S] { return !S->E->GetTrainingJob() || S->E->GetTrainingJob()->IsFinished(); }, 120.0);
	Step([this, S]
	{
		UPythonJob* Job = S->E->GetTrainingJob();
		if (!Job)
		{
			return;
		}
		AddInfo(FString::Printf(TEXT("train_reach: %s in %.1f s (%lld env steps)"), *Job->ResultJson, FPlatformTime::Seconds() - S->Started, Steps(S->E)));
		TestEqual(TEXT("train_reach completed: ") + Job->Error, Job->State, EPythonJobState::Completed);
		const float Mean = Job->GetMetric(TEXT("mean_return"));
		TestTrue(FString::Printf(TEXT("PPO reached the target return (mean %.2f over the last 100 episodes)"), Mean), Mean >= 1.2f);
		TestEqual(TEXT("the environment handed over to the trained policy (Run)"), S->E->Mode, ETorchEnvironmentMode::Run);
		S->RunEpisodesFrom = Episodes(S->E);
		Job->RemoveFromRoot();
	});
	Drive(this, TEXT("the trained policy running"), S->E, 4, [S] { return Episodes(S->E) >= S->RunEpisodesFrom + 150; }, 300.0);
	Step([this, S]
	{
		const float Mean = MeanReturn(S->E);
		AddInfo(FString::Printf(TEXT("trained policy (Run mode, published weights): mean return %.2f"), Mean));
		TestTrue(FString::Printf(TEXT("the published policy reaches its targets in Run mode (mean return %.2f)"), Mean), Mean > 1.2f);
		S->E->ResetWithSeed(42);
		S->From = S->A.LastStep = Steps(S->E);
	});
	//Deterministic replay: the same seed gives the same trajectory (twice), another seed a different one
	Drive(this, TEXT("run A (seed 42)"), S->E, 4, [S] { S->A.Sample(S->E); return Steps(S->E) >= S->From + 120; }, 300.0);
	Step([S] { S->E->ResetWithSeed(42); S->From = S->B.LastStep = Steps(S->E); });
	Drive(this, TEXT("run B (seed 42)"), S->E, 4, [S] { S->B.Sample(S->E); return Steps(S->E) >= S->From + 120; }, 300.0);
	Step([S] { S->E->ResetWithSeed(43); S->From = S->C.LastStep = Steps(S->E); });
	Drive(this, TEXT("run C (seed 43)"), S->E, 4, [S] { S->C.Sample(S->E); return Steps(S->E) >= S->From + 120; }, 300.0);
	Step([this, S]
	{
		const int32 N = FMath::Min(S->A.Positions.Num(), S->B.Positions.Num());
		TestTrue(FString::Printf(TEXT("runs sampled (%d, %d positions)"), S->A.Positions.Num(), S->B.Positions.Num()), N > 64 * 100);
		int32 Mismatch = 0;
		for (int32 i = 0; i < N; ++i)
		{
			Mismatch += S->A.Positions[i] == S->B.Positions[i] ? 0 : 1;
		}
		TestEqual(TEXT("same seed: the same trajectory, exactly"), Mismatch, 0);
		bool bDiffers = false;
		for (int32 i = 0; i < FMath::Min(N, S->C.Positions.Num()) && !bDiffers; ++i)
		{
			bDiffers = S->A.Positions[i] != S->C.Positions[i];
		}
		TestTrue(TEXT("another seed: another trajectory"), bDiffers);
		S->Env.Destroy();
	});
	return true;
}

#endif
