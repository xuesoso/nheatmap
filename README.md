![Logo](logo.png)
# Bag of STARs
STAR mapping on slurm clusters in bags. This fork enables modularization of configuration for execution parameters.

## Requirements
- `python3.4+`
- `STAR` in your `PATH`

## Installation
```bash
git clone https://github.com/iosonofabio/bag_of_stars.git
```

## Usage
Call bag of stars from the `bos` folder:
```bash
python bag_of_stars.py --genomeDir <your genome folder> --output <your output folder> --fastqFolder <your fastq root folder>
```
- The fastq root folder must contain subfolders with each a read1 and read2 file.
- New subfolders with the same names will be made inside the output folder.
- The genome folder must contain STAR's hash files (e.g. `SA`, `SAindex`, `Genome`)
- `STAR` must be in your `PATH`, you can check your `.bashrc` for what folders are there.

## Help
```bash
python bag_of_stars.py --help

usage: bag_of_stars.py [-h] [--dry] --output OUTPUT [-n N]
                       [--genomeDir GENOMEDIR] [--local]
                       [--cpus-per-task CPUS_PER_TASK] [--mem MEM]
                       [--time TIME]
                       fastq_folder

STAR mapping in bags

required arguments:
  --fastqFolder INPUT   Parent folder of subfolders with 2 fastq.gz files in
                        each.
  --output OUTPUT       Parent folder for the output. For each input
                        subfolder, an output subfolder will be made
  --genomeDir GENOMEDIR
                        Folder with the STAR genome hash

optional arguments:
  -h, --help            show this help message and exit
  --dry                 Dry run
  -n N                  Number of samples per STAR call
  --local               Do not send to cluster, do everything locally
  --cpus-per-task CPUS_PER_TASK
                        Number of CPUs for each STAR call
  --mem MEM             RAM memory in MB for each STAR call
  --time TIME           Time limit on each group of STAR jobs (see slurm docs
                        for format info)
  --partition PART      String with comma-separated list of partitions on
                        Sherlock (e.g. 'quake,owners,normal'). Use quotes
  --htseq               Also perform `htseq-count` to count features
  --annotationFile FN   Path to the GTF file for htseq
  --delete-empty-BAM    Delete empty BAM files and remap them
```
