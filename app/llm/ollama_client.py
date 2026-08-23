import requests


class OllamaClient:

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
    ) -> str:

        if not prompt or not prompt.strip():
            return ""

        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        try:

            response = requests.post(
                url,
                json=payload,
                timeout=120,
            )

            response.raise_for_status()

            data = response.json()

            return data.get("response", "").strip()

        except requests.exceptions.ConnectionError:

            raise RuntimeError(
                "Could not connect to Ollama. "
                "Make sure Ollama is running."
            )

        except requests.exceptions.Timeout:

            raise RuntimeError(
                "Ollama request timed out."
            )

        except requests.exceptions.RequestException as e:

            raise RuntimeError(
                f"Ollama request failed: {e}"
            )

        except Exception as e:

            raise RuntimeError(
                f"Unexpected Ollama error: {e}"
            )