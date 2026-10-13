#pragma once
// Receives Blueprint node delegates (Run Model's On Success / On Failure) in the automation tests.

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "TorchTypes.h"
#include "TorchTestReceiver.generated.h"

UCLASS(Transient)
class UTorchTestReceiver : public UObject
{
	GENERATED_BODY()
public:
	TArray<FTorchResult> Results;

	UFUNCTION()
	void OnResult(const FTorchResult& Result) { Results.Add(Result); }
};
