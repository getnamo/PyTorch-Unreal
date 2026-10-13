#pragma once
// Shared by the PyTorch automation tests. Run headless:
//   UnrealEditor-Cmd.exe <Project>.uproject -ExecCmds="Automation RunTests PyTorch; Quit" -unattended -nullrhi -nopause

#include "CoreMinimal.h"
#include "Misc/AutomationTest.h"
#include "PythonComponent.h"
#include "Engine/Engine.h"
#include "Engine/GameInstance.h"
#include "Engine/World.h"
#include "GameFramework/Actor.h"
#include "Tests/AutomationCommon.h"

namespace TorchTests
{
	constexpr EAutomationTestFlags Flags = EAutomationTestFlags::EditorContext | EAutomationTestFlags::ClientContext | EAutomationTestFlags::ProductFilter;
	/** Benchmarks: run on demand (RunTests PyTorch.Benchmarks), not with the product tests. */
	constexpr EAutomationTestFlags BenchmarkFlags = EAutomationTestFlags::EditorContext | EAutomationTestFlags::ClientContext | EAutomationTestFlags::PerfFilter;
	/** The first run installs torch and imports it (seconds) */
	constexpr double Timeout = 300.0;

	/** A game world for components (not ticked: tests drive what they need). */
	struct FEnv
	{
		UGameInstance* GameInstance = nullptr;
		UWorld* World = nullptr;
		TArray<TWeakObjectPtr<AActor>> Actors;

		void Create()
		{
			GameInstance = NewObject<UGameInstance>(GEngine);
			GameInstance->AddToRoot();
			GameInstance->InitializeStandalone();
			World = GameInstance->GetWorld();
			World->InitializeActorsForPlay(FURL());
		}

		AActor* Spawn()
		{
			AActor* Actor = World->SpawnActor<AActor>();
			Actors.Add(Actor);
			return Actor;
		}

		/** Configure runs before registration and BeginPlay, like editing the component's details. */
		template<typename T>
		T* Add(AActor* Actor, TFunctionRef<void(T*)> Configure)
		{
			T* C = NewObject<T>(Actor);
			Configure(C);
			C->RegisterComponent(); //Begins play itself once the actor has
			if (!Actor->HasActorBegunPlay())
			{
				Actor->DispatchBeginPlay();
			}
			return C;
		}

		void Destroy()
		{
			for (const TWeakObjectPtr<AActor>& A : Actors)
			{
				if (A.IsValid())
				{
					A->Destroy();
				}
			}
			GameInstance->Shutdown();
			GEngine->DestroyWorldContext(World);
			World->DestroyWorld(false);
			GameInstance->RemoveFromRoot();
		}
	};

	inline void Step(TFunction<void()> Fn)
	{
		ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand([Fn]() { Fn(); return true; }));
	}

	inline void WaitUntil(FAutomationTestBase* Test, const FString& Label, TFunction<bool()> Condition, double InTimeout = Timeout)
	{
		const TSharedRef<double> Deadline = MakeShared<double>(0.0);
		ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand([Test, Label, Condition, Deadline, InTimeout]() -> bool
		{
			if (*Deadline == 0.0)
			{
				*Deadline = FPlatformTime::Seconds() + InTimeout;
			}
			if (Condition())
			{
				return true;
			}
			if (FPlatformTime::Seconds() > *Deadline)
			{
				Test->AddError(TEXT("Timed out waiting for ") + Label);
				return true;
			}
			return false;
		}));
	}

	struct FCall
	{
		bool bDone = false;
		double Seconds = 0.0;
		FPythonCallResult Result;
	};

	inline TSharedRef<FCall> Call(UPythonComponent* C, const FString& Function, const FString& JsonArgs, const TArray<FPyTensor>& Tensors = {})
	{
		TSharedRef<FCall> Out = MakeShared<FCall>();
		const double Start = FPlatformTime::Seconds();
		C->CallFunction(Function, JsonArgs, Tensors, [Out, Start](const FPythonCallResult& R)
		{
			Out->Result = R;
			Out->Seconds = FPlatformTime::Seconds() - Start;
			Out->bDone = true;
		});
		return Out;
	}

	/** Percentile of a sorted array (P in 0..1). */
	inline double Percentile(const TArray<double>& Sorted, double P)
	{
		return Sorted.Num() ? Sorted[FMath::Clamp((int32)(P * (Sorted.Num() - 1) + 0.5), 0, Sorted.Num() - 1)] : 0.0;
	}
}
