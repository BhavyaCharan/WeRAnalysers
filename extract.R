library(data.table)

#Note:
#29 days in Feb for 2016
#2016-01-03 is the starting
#Modify the code  accordingly
for(year in (2016,2018))
{
  for (mon in (1:12))
  {
    if(mon ==2)
    {
      days=28
    }
    else if(mon %in% c(1,3,5,7,8,10,12))
    {
      days=31
    }
    else
    {
      days=30
    }
    if(nchar(toString(mon))==1)
    {
      month=paste("0",toString(mon),sep="")
    }
    else
    {
      month=toString(mon)
    }
    for (i in (1:days))
    {
      if(nchar(toString(i))==1)
      {
        day=paste("0",toString(i),sep="")
      }
      else
      {
        day=toString(i)
      }
      path_var= paste(toString(year),"/",toString(year),"-",month,"-",day,".csv",sep="")
      df=read.csv(path_var)
      df_india=df[df$country=="IN",]
      #df_year=paste(toString(year),"_","india.csv",sep="")
      print(path_var)
      #fwrite(df,"complete.csv",append = TRUE)
      fwrite(df,df_year,append = TRUE)
        
      
      
      #Write into df_year
      #Write into df_complete
      
    }
  }
}
