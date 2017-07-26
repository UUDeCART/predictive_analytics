format.percent <- 
function( x
        , places    = attr(x, "places")       #< Places to show after period
        , threshold = getOption("percent::threshold", -Inf)
        , ... #< currently ignored
        , justify = "right"
        ){
    #' @title format as a percent.
    #' 
    #' @param x           A numeric vector of raw percents
    #' @param places      Places to show after period
    #' @param threshold   minimum percent to show.
    #' @param ...         ignored
    #' @param justify     Justification
    #' 
    #' @description
    #'   formats a number with a percent sign.
    #'   Also will show a cutoff for very small percents.
    if (is.null(places)) places <- getOption("percent::places", 2)
    fmt <- paste0("%2.", places, "f%%")
    ifelse( x < threshold
          , sprintf("< %s", sprintf(fmt, threshold))
          , sprintf(fmt, x*100)
          )
    #! @return a formatted string.
}