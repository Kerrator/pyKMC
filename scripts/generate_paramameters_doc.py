import os
import sys

# Import DEFAULT, MANDATORY and DESCRIPTIONS from config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from pykmc.config import DEFAULT, MANDATORY, DESCRIPTIONS

def generate_parameters_md(output_file = "../docs/parameters.md") : 
    """Generate a Markdown file listing parameters, their default values and descriptions

    Parameters 
    ---------- 
    output_file : str, default '..docs/parameters.md'
        path to the output markdown file
    """    
    with open(output_file, 'w') as f : 
        f.write('# Inputs Parameters \n \n')

        for section, params in {**DEFAULT, **MANDATORY}.items() : 
            f.write(f"## {section}\n \n")
            for param in params : 
                default = DEFAULT.get(section, {}).get(param, "**MANDATORY**") #write Mandatory if no default value
                description = DESCRIPTIONS.get(section, {}).get(param, "No description available")
                f.write(f"**{param}** \n\n")
                f.write(f"- **Default**: `{default}` \n\n")
                f.write(f"- **Description**: {description} \n\n")

if __name__ == "__main__" : 
    generate_parameters_md()
