#include "TorchOutputComponent.h"
#include "PyTorch.h"
#include "PythonImageLibrary.h"
#include "PythonService.h"
#include "PythonStreamComponent.h"
#include "TorchModel.h"
#include "TorchSubsystem.h"
#include "Dom/JsonObject.h"
#include "Engine/TextureRenderTarget2D.h"
#include "GameFramework/Actor.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

UTorchOutputComponent::UTorchOutputComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
	bWantsInitializeComponent = true;
}

UPythonStreamComponent* UTorchOutputComponent::ResolveStream() const
{
	AActor* Owner = GetOwner();
	if (!Owner)
	{
		return nullptr;
	}
	TArray<UPythonStreamComponent*> Streams;
	Owner->GetComponents(Streams);
	for (UPythonStreamComponent* S : Streams)
	{
		if (StreamComponent.IsNone() || S->GetFName() == StreamComponent)
		{
			return S;
		}
	}
	return nullptr;
}

bool UTorchOutputComponent::SetTarget()
{
	UTorchSubsystem* Torch = UTorchSubsystem::Get();
	UPythonStreamComponent* Stream = ResolveStream();
	if (!Torch || !Model || !Stream || Method.IsEmpty())
	{
		UE_LOG(LogTorch, Warning, TEXT("%s: needs a Model, a Method and a Python stream component on the actor"), *GetPathName());
		return false;
	}
	UPythonService* Service = Torch->GetService(Model);
	if (!Service)
	{
		return false;
	}
	if (BindingId == 0)
	{
		BindingId = Torch->RegisterBinding(this);
	}
	TSharedRef<FJsonObject> Info = MakeShared<FJsonObject>();
	Info->SetStringField(TEXT("model"), Model->GetModelName());
	Info->SetStringField(TEXT("method"), Method);
	Info->SetStringField(TEXT("input"), InputName);
	Info->SetNumberField(TEXT("binding"), BindingId);
	TSharedPtr<FJsonObject> Extra;
	if (!JsonInputs.TrimStartAndEnd().IsEmpty() && FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(JsonInputs), Extra) && Extra.IsValid())
	{
		Info->SetObjectField(TEXT("inputs"), Extra);
	}
	Stream->SetTargetService(Service, TEXT("_torch_stream"), PythonWire::ToJsonString(MakeShared<FJsonValueObject>(Info)));
	Torch->LoadModel(Model); //With the asset's options, before the first frame
	return true;
}

void UTorchOutputComponent::InitializeComponent()
{
	Super::InitializeComponent();
	//Before the stream component's BeginPlay opens it: it opens straight on the model's service
	if (bBindOnBeginPlay && GetWorld() && GetWorld()->IsGameWorld())
	{
		SetTarget();
	}
}

void UTorchOutputComponent::BeginPlay()
{
	Super::BeginPlay();
	if (bBindOnBeginPlay)
	{
		UPythonStreamComponent* Stream = ResolveStream();
		if (Stream && !Stream->IsStreamOpen() && Stream->bOpenOnBeginPlay && Stream->HasBegunPlay())
		{
			Stream->OpenStream(); //The stream began play first and couldn't open without a target
		}
	}
}

bool UTorchOutputComponent::Bind()
{
	if (!SetTarget())
	{
		return false;
	}
	UPythonStreamComponent* Stream = ResolveStream();
	Stream->CloseStream();
	return Stream->OpenStream();
}

void UTorchOutputComponent::Unbind()
{
	if (UPythonStreamComponent* Stream = ResolveStream())
	{
		Stream->CloseStream();
	}
	if (UTorchSubsystem* Torch = UTorchSubsystem::Get(); Torch && BindingId)
	{
		Torch->UnregisterBinding(BindingId);
	}
	BindingId = 0;
}

void UTorchOutputComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	if (UTorchSubsystem* Torch = UTorchSubsystem::Get(); Torch && BindingId)
	{
		Torch->UnregisterBinding(BindingId);
	}
	BindingId = 0;
	Super::EndPlay(EndPlayReason);
}

float UTorchOutputComponent::GetResultsPerSecond() const
{
	const double Now = FPlatformTime::Seconds();
	int32 Count = 0;
	for (double T : RecentResults)
	{
		Count += Now - T <= 1.0 ? 1 : 0;
	}
	return (float)Count;
}

void UTorchOutputComponent::HandleResult(const FTorchResult& Result)
{
	const double Now = FPlatformTime::Seconds();
	RecentResults.Add(Now);
	RecentResults.RemoveAll([Now](double T) { return Now - T > 2.0; });
	++ResultCount;
	LastResult = Result;
	if (Result.bSuccess)
	{
		for (const TPair<FString, TObjectPtr<UTextureRenderTarget2D>>& Sink : RenderTargetSinks)
		{
			if (const FPyTensor* Image = Result.Outputs.Find(Sink.Key); Image && Sink.Value)
			{
				UPythonImageLibrary::WriteTensorToRenderTarget(*Image, Sink.Value, bChannelsFirst);
			}
		}
	}
	OnOutput.Broadcast(Result);
}
