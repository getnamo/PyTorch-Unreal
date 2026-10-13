#include "TorchReachExample.h"
#include "PythonJob.h"
#include "TorchModel.h"
#include "TorchSubsystem.h"
#include "Components/InputComponent.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Components/StaticMeshComponent.h"
#include "Components/TextRenderComponent.h"
#include "Engine/StaticMesh.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "Engine/World.h"
#include "GameFramework/PlayerController.h"
#include "InputCoreTypes.h"
#include "Modules/ModuleManager.h"
#include "UObject/ConstructorHelpers.h"

IMPLEMENT_MODULE(FDefaultModuleImpl, PyTorchExamples)

// ------------------------------------------------------------------------------------------------- the agent

void UTorchReachAgentComponent::CollectObservations()
{
	SetObservations(TEXT("to_target"), { (float)((Target.X - Position.X) / ArenaHalfSize), (float)((Target.Y - Position.Y) / ArenaHalfSize) });
	SetObservations(TEXT("velocity"), { (float)(Velocity.X / MaxSpeed), (float)(Velocity.Y / MaxSpeed) });
	Super::CollectObservations();
}

void UTorchReachAgentComponent::ApplyActions(const TArray<float>& InActions)
{
	const float Dt = GetStepSeconds();
	const FVector2D Command(InActions.IsValidIndex(0) ? FMath::Clamp(InActions[0], -1.f, 1.f) : 0.f, InActions.IsValidIndex(1) ? FMath::Clamp(InActions[1], -1.f, 1.f) : 0.f);
	const double Before = FVector2D::Distance(Position, Target);
	Velocity = Velocity * 0.5 + Command * MaxSpeed * 0.5; //A little inertia
	Position += Velocity * Dt;
	Position.X = FMath::Clamp(Position.X, -ArenaHalfSize, ArenaHalfSize);
	Position.Y = FMath::Clamp(Position.Y, -ArenaHalfSize, ArenaHalfSize);
	const double After = FVector2D::Distance(Position, Target);
	AddReward((float)((Before - After) / ArenaHalfSize) - 0.01f);
	if (After < ReachRadius)
	{
		AddReward(1.f);
		EndEpisode(true);
	}
	UpdateVisuals();
	Super::ApplyActions(InActions);
}

void UTorchReachAgentComponent::ResetAgent()
{
	Position = FVector2D(RandomRange(-ArenaHalfSize, ArenaHalfSize), RandomRange(-ArenaHalfSize, ArenaHalfSize));
	do
	{
		Target = FVector2D(RandomRange(-ArenaHalfSize, ArenaHalfSize), RandomRange(-ArenaHalfSize, ArenaHalfSize));
	} while (FVector2D::Distance(Position, Target) < ArenaHalfSize * 0.5f);
	Velocity = FVector2D::ZeroVector;
	UpdateVisuals();
	Super::ResetAgent();
}

void UTorchReachAgentComponent::Heuristic()
{
	const FVector2D Direction = (Target - Position).GetSafeNormal();
	SetHeuristicActions({ (float)Direction.X, (float)Direction.Y });
	Super::Heuristic();
}

void UTorchReachAgentComponent::UpdateVisuals()
{
	if (ATorchReachAgent* Actor = Cast<ATorchReachAgent>(GetOwner()))
	{
		Actor->Body->SetRelativeLocation(FVector(Position.X, Position.Y, 30.0));
		Actor->TargetMarker->SetRelativeLocation(FVector(Target.X, Target.Y, 15.0));
	}
}

// ------------------------------------------------------------------------------------------------- the actor

ATorchReachAgent::ATorchReachAgent()
{
	PrimaryActorTick.bCanEverTick = false;
	static ConstructorHelpers::FObjectFinder<UStaticMesh> Sphere(TEXT("/Engine/BasicShapes/Sphere.Sphere"));
	static ConstructorHelpers::FObjectFinder<UStaticMesh> Cube(TEXT("/Engine/BasicShapes/Cube.Cube"));
	static ConstructorHelpers::FObjectFinder<UStaticMesh> Plane(TEXT("/Engine/BasicShapes/Plane.Plane"));
	static ConstructorHelpers::FObjectFinder<UMaterialInterface> ShapeMaterial(TEXT("/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial")); //Has a Color parameter
	RootComponent = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
	auto MakeMesh = [this, Material = ShapeMaterial.Object](const TCHAR* Name, UStaticMesh* Mesh, const FVector& Scale)
	{
		UStaticMeshComponent* C = CreateDefaultSubobject<UStaticMeshComponent>(Name);
		C->SetupAttachment(RootComponent);
		C->SetStaticMesh(Mesh);
		C->SetMaterial(0, Material);
		C->SetRelativeScale3D(Scale);
		C->SetCollisionEnabled(ECollisionEnabled::NoCollision);
		return C;
	};
	Floor = MakeMesh(TEXT("Floor"), Plane.Object, FVector(6.4, 6.4, 1.0)); //The arena: 640 units (Arena Half Size 300, plus a margin)
	Body = MakeMesh(TEXT("Body"), Sphere.Object, FVector(0.6));
	TargetMarker = MakeMesh(TEXT("Target"), Cube.Object, FVector(0.3));
	Agent = CreateDefaultSubobject<UTorchReachAgentComponent>(TEXT("Agent"));
}

void ATorchReachAgent::PostInitializeComponents()
{
	Super::PostInitializeComponents(); //(Not BeginPlay: agents spawned during the world's BeginPlay may begin play late)
	//BasicShapeMaterial's Color: a dark arena, an orange agent, a green target
	const TPair<UStaticMeshComponent*, FLinearColor> Colors[] = { { Floor, FLinearColor(0.03f, 0.035f, 0.045f) }, { Body, FLinearColor(1.f, 0.3f, 0.05f) }, { TargetMarker, FLinearColor(0.1f, 0.9f, 0.2f) } };
	for (const TPair<UStaticMeshComponent*, FLinearColor>& C : Colors)
	{
		if (UMaterialInstanceDynamic* M = C.Key->CreateAndSetMaterialInstanceDynamic(0))
		{
			M->SetVectorParameterValue(TEXT("Color"), C.Value);
		}
	}
}

// ------------------------------------------------------------------------------------------- the environment

ATorchReachEnvironment::ATorchReachEnvironment()
{
	EnvName = TEXT("Reach");
	Observations = { { TEXT("to_target"), 2 }, { TEXT("velocity"), 2 } };
	Actions.Type = ETorchActionType::Continuous;
	Actions.Size = 2;
	MaxEpisodeSteps = 200;
	StepSeconds = 0.05f;
	TrainingJob = TEXT("train_reach");
	RecordingName = TEXT("Reach");
	StatusText = CreateDefaultSubobject<UTextRenderComponent>(TEXT("StatusText"));
	StatusText->SetupAttachment(RootComponent);
	StatusText->SetHorizontalAlignment(EHTA_Left);
	StatusText->SetWorldSize(50.f);
	StatusText->SetTextRenderColor(FColor(120, 255, 120));
	StatusText->SetRelativeLocation(FVector(-700.0, -1100.0, 150.0)); //In front of the arenas, facing the example map's camera
	StatusText->SetRelativeRotation(FRotator(0.0, 180.0, 0.0));
	static ConstructorHelpers::FObjectFinder<UMaterialInterface> UnlitText(TEXT("/Engine/EngineMaterials/UnlitText.UnlitText"));
	StatusText->SetTextMaterial(UnlitText.Object); //Readable without lights
	AutoReceiveInput = EAutoReceiveInput::Player0;
}

void ATorchReachEnvironment::BeginPlay()
{
	SpawnAgents();
	Super::BeginPlay();
	StatusText->SetVisibility(bExampleControls);
}

void ATorchReachEnvironment::Tick(float DeltaSeconds)
{
	Super::Tick(DeltaSeconds);
	StatusRefresh -= DeltaSeconds;
	if (bExampleControls && StatusRefresh <= 0.f)
	{
		StatusRefresh = 0.25f; //A text change rebuilds the text's render state: a few times a second is plenty
		StatusText->SetText(FText::FromString(GetStatus()));
	}
}

void ATorchReachEnvironment::EnableInput(APlayerController* PlayerController)
{
	Super::EnableInput(PlayerController);
	if (!bExampleControls || bKeysBound || !InputComponent)
	{
		return;
	}
	bKeysBound = true;
	for (const FKey& Key : { EKeys::T, EKeys::R, EKeys::H, EKeys::C, EKeys::B, EKeys::One, EKeys::Two })
	{
		InputComponent->BindKey(Key, IE_Pressed, this, &ATorchReachEnvironment::OnKey);
	}
}

void ATorchReachEnvironment::OnKey(FKey Key)
{
	if (Key == EKeys::T) { SetMode(ETorchEnvironmentMode::Train); bShowCloning = false; }
	else if (Key == EKeys::R) { SetMode(ETorchEnvironmentMode::Run); }
	else if (Key == EKeys::H) { SetMode(ETorchEnvironmentMode::Heuristic); }
	else if (Key == EKeys::C) { SetMode(ETorchEnvironmentMode::Record); }
	else if (Key == EKeys::B) { StartBehaviorCloning(); }
	else if (Key == EKeys::One) { ResetWithSeed(1); }
	else if (Key == EKeys::Two) { ResetWithSeed(2); }
}

UPythonJob* ATorchReachEnvironment::StartBehaviorCloning()
{
	UTorchSubsystem* Torch = UTorchSubsystem::Get();
	UTorchModel* Model = PolicyModel ? PolicyModel.Get() : TrainingModel.Get();
	if (!Torch || !Model || (CloningJob && !CloningJob->IsFinished()))
	{
		return CloningJob;
	}
	SetMode(ETorchEnvironmentMode::Off); //Ends (flushes) a recording
	bShowCloning = true;
	CloningJob = Torch->StartJob(Model, TEXT("train_reach_bc"), FString::Printf(TEXT("{\"recording\": \"%s\", \"sessions\": \"latest\"}"), *RecordingName), false, true);
	return CloningJob;
}

FString ATorchReachEnvironment::GetStatus() const
{
	static const TCHAR* Names[] = { TEXT("Off"), TEXT("Train"), TEXT("Run"), TEXT("Heuristic (scripted expert)"), TEXT("Record (scripted expert)") };
	int64 Steps;
	int32 Count;
	float Mean;
	GetStats(Steps, Count, Mean);
	FString Status = FString::Printf(TEXT("Reach: %s\nsteps %lld, episodes %d, mean return %.2f"), Names[FMath::Clamp((int32)Mode, 0, 4)], Steps, Count, Mean);
	const UPythonJob* Job = bShowCloning && CloningJob ? CloningJob.Get() : GetTrainingJob();
	if (Job)
	{
		FString Line = Job->ProgressMessage;
		if (Job->IsFinished())
		{
			TSharedPtr<FJsonObject> Result;
			Line = !Job->Error.IsEmpty() ? TEXT("Failed: ") + Job->Error
				: FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Job->ResultJson), Result) && Result.IsValid() && Result->HasField(TEXT("text")) ? Result->GetStringField(TEXT("text")) : TEXT("done");
		}
		Status += TEXT("\n") + Job->JobName + TEXT(": ") + Line.Left(90);
	}
	return Status + TEXT("\n[T] train  [R] run  [H] expert  [C] record  [B] clone  [1]/[2] replay a seed");
}

void ATorchReachEnvironment::SpawnAgents()
{
	if (bSpawned || !GetWorld())
	{
		return;
	}
	bSpawned = true;
	const int32 Columns = FMath::CeilToInt(FMath::Sqrt((float)NumAgents));
	for (int32 i = 0; i < NumAgents; ++i)
	{
		const FVector Location = GetActorLocation() + FVector((i / Columns) * Spacing, (i % Columns - (Columns - 1) * 0.5) * Spacing, 0.0);
		//Deferred: the agent knows its environment before its BeginPlay (which registers it with the first one)
		ATorchReachAgent* A = GetWorld()->SpawnActorDeferred<ATorchReachAgent>(ATorchReachAgent::StaticClass(), FTransform(Location), this, nullptr, ESpawnActorCollisionHandlingMethod::AlwaysSpawn);
		if (A)
		{
			A->Agent->Environment = this;
			RegisterAgent(A->Agent); //Now, in spawn order (its BeginPlay registration is then a no-op)
			A->FinishSpawning(FTransform(Location));
		}
	}
}
