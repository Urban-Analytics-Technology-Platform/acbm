folderIn <- "data"

readNTS <- function(name,folderIn = "data"){
  data <- read.table(file.path(folderIn,"UKDA-5340-tab","tab",paste0(name,"_eul_2002-2022.tab")), sep="\t", header=TRUE)
  return(data)
}

indiv <- readNTS("individual",folderIn)
stage <- readNTS("stage",folderIn)
trip <- readNTS("trip",folderIn)

trip2022 <- trip[trip$SurveyYear == 2022,]
stage2022 <- stage[stage$SurveyYear == 2022,]

# Missing values
neg <- which(trip2022 < 0, arr.ind = T)
nrow(neg)
unique(colnames(trip2022)[neg[,2]])
length(which(neg[,2] == neg[,2][1]))/nrow(trip2022)

negS <- which(stage2022 < 0, arr.ind = T)
nrow(negS)
unique(colnames(stage2022)[negS[,2]])
length(which(negS[,2] == negS[,2][1]))/nrow(stage2022)
