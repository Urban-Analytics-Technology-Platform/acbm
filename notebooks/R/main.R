folderIn <- "data"

readNTS <- function(name,folderIn = "data"){
  data <- read.table(file.path(folderIn,"UKDA-5340-tab","tab",paste0(name,"_eul_2002-2022.tab")), sep="\t", header=TRUE)
  return(data)
}

NTS <- readNTS("individual",folderIn)
stage <- readNTS("stage",folderIn)
trip <- readNTS("trip",folderIn)