# Segunda revision tecnica

Fecha: 2026-07-09

Alcance: paquete `ollama_client`, tests locales, metadata de proyecto y compatibilidad
del contrato publico ya graduado. No se cambiaron tecnologias ni dependencias de
runtime.

## Hallazgos corregidos

| Ubicacion | Problema | Impacto | Sugerencia / cambio aplicado |
|---|---|---|---|
| `ollama_client/_transport.py:_post` | Una respuesta HTTP 200 con JSON invalido escapaba como `json.JSONDecodeError`, fuera de la taxonomia `OllamaRequestError` / `OllamaUnavailable`. | Los fallback chains podian abortar con una excepcion cruda en vez de probar el siguiente modelo o degradar de forma controlada. | `_post` ahora convierte JSON invalido o JSON no objeto en `OllamaRequestError(502, ...)`. Tests: `test_post_invalid_json_becomes_request_error`, `test_post_non_object_json_becomes_request_error`. |
| `ollama_client/generation.py` y `ollama_client/chat.py` | Las entradas de cache se escribian directamente con `Path.write_text`. | Si el proceso moria durante la escritura, otro consumidor podia leer una entrada parcial o corrupta. | Se agrego `_write_cache_text()` en `_cache.py`, usando archivo temporal en el mismo directorio y `replace()`. `generate` y `chat` lo usan. Test: `test_write_cache_text_replaces_without_tmp_leftovers`. |
| `ollama_client/_version.py:_parse_version` | El parser concatenaba todos los digitos de cada segmento. Ejemplo previo: `1.0.0-rc1` -> `(1, 0, 1)`, contradiciendo el docstring de ignorar prerelease. | Un gate `require()` podia comparar mal versiones con sufijos y fallar o pasar por razones equivocadas. | `_parse_version` ahora toma el primer grupo numerico por segmento. Tests actualizados para `1.0.0-rc1` y `1.2rc3`. |
| `tests/conftest.py:fake_urlopen` | El fixture solo podia devolver dicts serializados como JSON valido. | No permitia cubrir respuestas malformadas del daemon, justo el caso de borde del transporte. | Se agrego `set_raw_response(bytes)` para pruebas de transporte sin red. |

## Hallazgos ya corregidos en la pasada anterior

| Ubicacion | Problema | Impacto | Estado |
|---|---|---|---|
| `pyproject.toml` | `pytest` desde checkout no veia el paquete si no estaba instalado editable. | Reproducibilidad pobre para colaboradores y workers. | `pythonpath = ["."]`. |
| `ollama_client/__init__.py:__all__` | `__all__` no incluia constantes publicas ni `main`. | `from ollama_client import *` no era equivalente al modulo plano anterior. | `__all__` expandido y cubierto por tests. |
| `ollama_client/cli.py` | `is-alive` no aceptaba `--base-url`, aunque el dispatcher intentaba leerlo. | CLI inconsistente frente a `generate`, `embed` y `ocr-image`. | `is-alive --base-url` agregado y probado. |
| `ollama_client/_transport.py` | `base_url` aceptaba esquemas arbitrarios antes de llamar a `urlopen`. | Superficie innecesaria para URLs no Ollama. | Normalizacion centralizada y solo `http/https`; SAST limpio con supresion precisa para la URL dinamica esperada. |
| `~/.claude/scripts/model-drift-check.py` | El checker parseaba constantes por regex en el shim graduado, donde ya no existen literales. | Reportaba drift falso para todos los defaults de `ollama_client`. | El checker importa `ollama_client` y lee atributos reales. |

## Oportunidades no aplicadas

| Ubicacion | Problema | Impacto | Sugerencia |
|---|---|---|---|
| `ollama_client/generation.py` y `ollama_client/chat.py` | Hay duplicacion entre lectura/escritura de cache en ambos dominios. | Mantenimiento algo mas costoso si cambia la politica de cache. | Extraer un helper de lectura de cache solo si aparece otro tercer consumidor; con dos sitios, el cambio no compensa todavia el riesgo sobre el contrato congelado. |
| `ollama_client/_transport.py:_normalize_base_url` | Actualmente rechaza `localhost:11434` sin esquema. | Es estricto y seguro, pero menos tolerante con CLI manual. | Mantenerlo asi por ahora: todos los consumidores revisados usan `http://...`, y aceptar esquemas implicitos podria ocultar configuracion ambigua. |
| `ollama_client/vision.py` | `ocr_image` no cachea OCR. | OCR repetido puede ser costoso. | No aplicar sin decision explicita: las salidas OCR pueden depender de modelo/prompt/opciones y el flujo PDF puede preferir frescura. |

## Validacion

- `pytest` -> 89 tests passed.
- `ruff check .` -> passed.
- `ruff format --check .` -> passed.
- `pyright` -> 0 errors.
- `codescan` de secretos, SAST y dead-code -> 0 hallazgos.
- Smokes de CLI/shim: `python3 -m ollama_client --version`, `python3 ~/.claude/scripts/ollama_client.py --version`.

## Conclusion

El proyecto queda mas robusto en tres areas criticas para un cliente compartido:
errores de transporte quedan dentro de la taxonomia publica, la cache evita
entradas parciales, y el gate SemVer compara sufijos de forma coherente con su
documentacion.
