# Documentation 

For the documentation, we use MkDocs. To install it with the material theme :
```bash 
pip install mkdocs mkdocstrings[python] mkdocs-material
```
The documentation is in the `./docs/`folder. 
List of markdown files : 
- index.md : homepage
- installation.md : how to install pyKMC
- troubleshooting.md : troublesshootings 
- tutorials.md : tutorials
- reference.md : API 

In the pyKMC folder, the `mkdocs.yml` deals with the MkDocs configuration. 

The `.github/workflows/deploy_docs.yml` file deal with the autogeneration of the documentation at each commit on the main branch.