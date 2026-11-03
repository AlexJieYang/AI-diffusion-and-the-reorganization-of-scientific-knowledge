pkgs <- c('IRkernel','fixest','arrow','data.table','jsonlite')
project <- normalizePath(getwd(),winslash='/',mustWork=TRUE)
jbin <- file.path(project,'.venv',if(.Platform$OS.type=='windows') 'Scripts' else 'bin')
if(dir.exists(jbin)) Sys.setenv(PATH=paste(jbin,Sys.getenv('PATH'),sep=.Platform$path.sep))
missing <- pkgs[!vapply(pkgs,requireNamespace,logical(1),quietly=TRUE)]
if(length(missing)) install.packages(missing,repos='https://cloud.r-project.org')
IRkernel::installspec(user=TRUE,name='ir-ai-atlas',displayname='R - AI Science Atlas')
