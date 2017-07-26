sift <- 
function( x                 #< A character vector 
        , pat               #< Regular expression pattern
        , value=TRUE        #< Return value or indices.
        , perl=TRUE         #< is `pat` perl compatible?
        , ignore.case=TRUE  #< Should case be ignored?
        , ...               #< passed on to <grep>
        ){
    #' Pipe compatible regular expression filtering.
    unique(grep( pattern=pat, x=x
               , value=value, ignore.case=ignore.case, perl=perl
               , ...))
    #' @seealso <grep>.
    #' @return A subset of x vector that matches the pattern given in `pat`.
}
sieve <- function(...)sift(..., invert=TRUE)