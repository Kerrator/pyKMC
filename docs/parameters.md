# Configuration Parameters

Reference for pyKMC INI sections and fields, generated from the Pydantic
models in `pykmc/config.py`. Each **section** corresponds to a
`[SECTION_NAME]` in your INI configuration file, and the listed parameters
are the fields available within that section. Section names are
case-insensitive; field names are case-sensitive. A section marked
"conditionally required" must be present when the controlling style or
feature is enabled.

{! ./docs/parameters_details.md !}
