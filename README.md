# Viral-Genomes

This project is split into two notebooks, Viral Genomes Data Acquisition and Viral Genomes and a variety of .py files.

Viral Genomes Data Acquisition.ipynb is the second iteration of the data collection code, which produces several csv files, including 'final_viral_genomes.csv', which is imported in the Viral Genomes notebook. Most of these csv files are too large to upload normally to github, hence their absence in the repo. 
    Note- the Entrez portion of the Acquisition notebook should be run outside of 5A.M.-9P.M. in order to avoid overloading the system during business hours. Also, be sure to update the email to the user's email.

Viral Genomes.ipynb contains data examination/cleaning, EDA, preprocessing, and modelling with Convoluted Neural Networks. Much of the preprocessing and modelling was done in the terminal either on my laptop or an AWS EC2 instance. 

The various model .py files were run and modified directly in the terminal after several jupyter notebook crashes, and their outputs are included in the repository as screenshots. When I have the chance I will continue to improve these models and upload the best .h5 file for each model. 

The Viral Genome Report includes a complete breakdown of my process in an informal report.

Moving forward, my next steps will be to improve models 0 and 5, and to attempt classifying the viruses which are listed as unclassified in the NCBI database. 
