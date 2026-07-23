import importlib.util
import sys
import os
from typing import get_args, get_origin, Union, Optional, Any, Literal
from pydantic import BaseModel, Field
import inspect

_spec = importlib.util.spec_from_file_location(
    "pykmc.config",
    os.path.join(os.path.dirname(__file__), "..", "pykmc", "config.py"),
)
_config_module = importlib.util.module_from_spec(_spec)
sys.modules["pykmc.config"] = _config_module
_spec.loader.exec_module(_config_module)
Config = _config_module.Config


def format_type(field_type: Any) -> str:
    """Formats a Python type hint into a readable string."""
    origin = get_origin(field_type)
    args = get_args(field_type)

    if origin is None:
        # Handle simple types (str, int, float, bool, etc.)
        return (
            field_type.__name__ if hasattr(field_type, "__name__") else str(field_type)
        )
    elif origin is Union:
        # Handle Optional[X] and Union[X, Y]
        formatted_args = []
        for arg in args:
            if arg is not type(None):  # Exclude NoneType for Optional
                formatted_args.append(format_type(arg))
        return " or ".join(formatted_args)
    elif origin is Literal:
        # Handle Literal['a', 'b']
        return f"Literal[{', '.join(repr(a) for a in args)}]"
    elif hasattr(origin, "__name__"):
        # Handle generic types like list, dict, tuple, etc.
        if args:  # If there are generic arguments (e.g., List[str])
            return f"{origin.__name__}[{', '.join(format_type(a) for a in args)}]"
        else:  # If it's just a generic type without specific arguments (e.g., list)
            return origin.__name__
    else:
        return str(field_type)


# Canonical INI spellings for the section headings (the Config attribute names
# are lowercase; the site and user guide use these spellings).
SECTION_DISPLAY_NAMES = {
    "control": "Control",
    "atomicenvironment": "AtomicEnvironment",
    "eventsearch": "EventSearch",
    "rateconstant": "RateConstant",
    "psr": "PSR",
    "lammps": "LAMMPS",
    "partn": "pARTn",
    "ira": "IRA",
    "reconstruction": "Reconstruction",
    "basin": "Basin",
    "activevolume": "ActiveVolume",
    "eventrecycling": "EventRecycling",
    "inactive_atoms": "Inactive_Atoms",
    "frozen_atoms": "Frozen_Atoms",
    "bias": "Bias",
}

# Sections whose mere presence is enforced when the INI file is parsed
# (Config.from_ini_file).
ALWAYS_REQUIRED_SECTIONS = {
    "control",
    "atomicenvironment",
    "eventsearch",
    "rateconstant",
    "psr",
}

# Sections enforced by Config.validate_dependencies when the controlling
# field takes the given value.
CONDITIONALLY_REQUIRED_SECTIONS = {
    "lammps": "required when `[Control]` `engine = lammps`",
    "partn": "required when `[EventSearch]` `style = partn`",
    "ira": "required when `[PSR]` `style = ira`",
    "basin": "required when `[Control]` `basin = True`",
    "activevolume": "required when `[Control]` `active_volume = True`",
    "eventrecycling": "required when `[Control]` `recycle = True`",
    "bias": "required when `[Control]` `bias = True`",
}


def is_required_pydantic_field(field: Field) -> bool:
    """Checks if a Pydantic field is mandatory."""
    return field.is_required()


def is_optional_pydantic_field(field_type: Any) -> bool:
    """Checks if a Pydantic field's type annotation indicates it's Optional."""
    origin = get_origin(field_type)
    return origin is Union and type(None) in get_args(field_type)


def section_status(section_name: str) -> str:
    """Return the requirement status string for a top-level section."""
    if section_name in ALWAYS_REQUIRED_SECTIONS:
        return "mandatory"
    if section_name in CONDITIONALLY_REQUIRED_SECTIONS:
        return (
            f"conditionally required — {CONDITIONALLY_REQUIRED_SECTIONS[section_name]}"
        )
    return "optional"


def document_model(
    model_class: type[BaseModel],
    section_name: str,
    status: str,
    parent_description: str | None = None,
) -> str:
    """
    Generates Markdown documentation for a Pydantic BaseModel section.

    Parameters
    ----------
    model_class : type[BaseModel]
        The Pydantic BaseModel class to document.
    section_name : str
        The Config attribute name of the section (e.g., 'control', 'lammps').
    status : str
        Requirement status of the section (mandatory / conditionally
        required / optional).
    parent_description : str, optional
        The description of the section's field on the parent Config model,
        shown before the model's own docstring (distinguishes sections that
        share a model class, e.g. the two Region sections).

    Returns
    -------
    str
        A Markdown formatted string for the section.
    """
    lines = []

    display_name = SECTION_DISPLAY_NAMES.get(section_name, section_name.capitalize())

    # Section heading with a stable anchor for deep links
    lines.append(f'<a id="section-{section_name}"></a>\n')
    lines.append(f"## `{display_name}` Section ({status})\n")

    # Overview: the parent Config field description first (it distinguishes
    # sections sharing a model class), then the class's own docstring.
    overview_parts = []
    class_doc = inspect.getdoc(model_class)
    if parent_description and parent_description.strip() != (class_doc or "").strip():
        overview_parts.append(parent_description)
    if class_doc:
        overview_parts.append(class_doc)
    if overview_parts:
        overview_indented = "\n  ".join("\n\n".join(overview_parts).splitlines())
        lines.append(
            f"<details><summary>Section Overview</summary>\n  {overview_indented}\n</details>\n"
        )
    else:
        lines.append("No overview description provided for this section.\n")

    for name, field in model_class.model_fields.items():
        typ = format_type(field.annotation)

        # Determine status (mandatory, optional with default, optional without default)
        status_info = ""
        if is_required_pydantic_field(field):
            status_info = "mandatory"
        elif field.default_factory is not None:
            status_info = f"default = `{field.default_factory()!r}`"
        elif is_optional_pydantic_field(field.annotation) and field.default is None:
            status_info = "optional"
        elif field.default is not None:
            status_info = f"default = `{field.default!r}`"
        else:
            status_info = "optional"

        # Parameter line (bullet point) with a stable anchor for deep links
        lines.append(
            f'- <a id="{section_name}-{name}"></a>**`{name}`** : `{typ}`, {status_info}'
        )

        # Description as a clickable <details> block
        desc = field.description or "No description provided."
        # Indent the description for Markdown details block
        desc_indented = "\n  ".join(desc.splitlines())
        lines.append(
            f"  <details><summary>Description</summary>\n  {desc_indented}\n  </details>"
        )

    lines.append("\n---\n")  # Add a separator at the end of each section
    return "\n".join(lines)


def generate_parameters_md(
    config_class: type[BaseModel],
    output_file: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "docs",
        "parameters_details.md",
    ),
):
    """
    Generates a Markdown file detailing all configuration parameters from a Pydantic Config model.

    Parameters
    ----------
    config_class : type[BaseModel]
        The root Pydantic BaseModel class (e.g., `Config`) to document.
    output_file : str
        The full path to the output Markdown file. Defaults to `docs/parameters_details.md`.
    """
    lines = []

    for name, field in config_class.model_fields.items():
        field_type = field.annotation

        # Determine if the top-level section (like 'lammps') is Optional
        is_top_level_optional = is_optional_pydantic_field(field_type)

        # Get the actual BaseModel class if it's Optional[BaseModel] or direct BaseModel
        sub_model_class = None
        if inspect.isclass(field_type) and issubclass(field_type, BaseModel):
            sub_model_class = field_type
        elif is_top_level_optional:
            # Find the BaseModel class among the Union args (excluding NoneType)
            sub_model_class = next(
                (
                    arg
                    for arg in get_args(field_type)
                    if inspect.isclass(arg) and issubclass(arg, BaseModel)
                ),
                None,
            )

        if sub_model_class:
            lines.append(
                document_model(
                    sub_model_class,
                    name,
                    section_status(name),
                    parent_description=field.description,
                )
            )
        # If your top-level Config also has direct fields (not sub-models), you'd handle them here.
        # For example, if Config had a `version: str` field directly.
        # Otherwise, pass.
        else:
            pass  # No direct fields in Config that aren't sub-models to document in this style.

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    generate_parameters_md(Config)
