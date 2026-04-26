### Overview
CSC supercomputers (Puhti, Mahti) use Linux command-line environments. Instead of GUI, work is done via terminal, which is:
    - Faster for repetitive tasks
    - Scriptable & automatable
    - Essential for HPC workflows
### Basic Usage
#### Connect via SSH
    ssh myusername@puhti.csc.fi

- Default shell: bash
- Use Tab for auto-completion

### Files & Directories
#### Key directories
- `/home` – personal files
- `/scratch` – fast temporary storage
- `/projappl` – shared project data

#### Navigation commands
```
pwd          # current directory
ls -l        # list files (detailed)
ls -a        # include hidden files
cd folder/   # enter directory
cd ..        # go up
```
#### Command structure
`command [options] [arguments]`


### Special Characters

- `*` – wildcard
- `?` – single character
- `~` – home directory
- `.` / `..` – current / parent
- `\` – escape character


### Editing Files
#### Editors
```
nano file.sh      # beginner-friendly
vim file.sh       # advanced
emacs -nw file.sh # terminal mode
```
- nano shortcuts: `Ctrl+O`, `Ctrl+X`
- vim: `i` → edit, `Esc :wq` → save & quit


### Process Control
```
Ctrl + C   # stop program
Ctrl + Z   # suspend
jobs       # list jobs
bg         # run in background
fg         # bring to foreground
```
#### Run in background:
```
script.sh &`
```

### File Management
#### Find & inspect
```
find . -name "*.txt"
find . -mtime -7
file data.dat
```
#### Count & compare
```
wc -l file.txt
diff old.txt new.txt
```
#### Checksums
```
md5sum file
sha256sum file
```
#### Encryption
```
gpg -c secret.txt
gpg -d secret.txt.gpg > secret.txt
```


### Permissions
```
chmod u+x script.sh
chmod 755 folder
```
- `r` = read, `w` = write, `x` = execute
- Important for shared HPC systems


### Data Processing
#### Core tools
```
grep "ERROR" file.log
sed -n '10,20p' file
cut -d',' -f1,3 data.csv
awk '{print $1}' file
sort file
uniq
tr 'A-Z' 'a-z'
```
#### Example pipeline
```
grep "Error" log.txt | cut -d' ' -f4 | sort | uniq -c | sort -nr | head -5
```
- **Key idea:** Process huge datasets without Python

### Compression & Archiving
#### tar
```
tar -cvf archive.tar folder/
tar -xvf archive.tar
```
#### Compression
```
gzip file
bzip2 file
zip archive.zip file
```
#### Combined
```
tar -czf archive.tar.gz folder/
tar -xzf archive.tar.gz
```

### Bash Scripting
#### Basic script
```
#!/bin/bash
echo "Hello"
```
#### Run:
```
bash script.sh
chmod +x script.sh && ./script.sh
```
#### Variables & arrays
```
name="Puhti"
echo $name
```
```
arr=(one two three)
echo ${arr[1]}
```
#### Loops & conditions
```
for file in *.txt; do
    wc -l $file
done
```
```
if [ -f file ]; then
    echo "Exists"
fi
```
#### Output
```
echo "Text"
printf "%05d\n" 42
```
### Example Script
```
#!/bin/bash
find /scratch/project/logs -name "*.log" -mtime +7 -exec gzip {} \;
echo "Done at $(date)" >> report.txt
```