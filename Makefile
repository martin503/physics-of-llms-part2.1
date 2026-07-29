
help: ## Show help
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

test: ## Run tests
	uv run pytest

fast-test: ## Run tests that are not slow
	uv run pytest -m "not slow"

test-slow: ## Run only slow tests
	uv run pytest -m "slow"

pre-commit: ## Install pre commit hooks
	uv run pre-commit install
	uv run pre-commit install-hooks

format: ## Format with pre commit
	uv run pre-commit run --all-files

