options(repos="https://cran.rstudio.com/")

is_r_startup <- function(){
    #! determing if R is starting up.
    root <- sys.call(1)
    !is.null(root) && (deparse(root) == ".First.sys")
}

using <- 
function( ...   #< Packages to load 
        , install.missing=!is.null(getOption("repos"))
                #< Should an attempt be made to install missing packages?
        , warn.conflicts = getOption("warn.conflict", FALSE)  #< not the default for library
        , quietly = FALSE
        , verbose = getOption("verbose")
        ){
    #! load library, install if missing.
    #! 
    #! Will attempt to install missing packages if the `repos` option is set.
    #! can also be used in .Rprofile files through `short::using`
    #! to add package to the default packages to be loaded at startup.
    libs <- as.character(substitute(c(...)))[-1]
    if(is_r_startup()){
        . <- libs %in% .packages(TRUE)
        if(any(!.)){
            warning("Packages", and_list(libs[!.]), " are not installed.")
            libs <- libs[.]
        }
        options(defaultPackages = c(getOption("defaultPackages"), libs))
        invisible(FAlSE)
    } else {
        is.loaded <- suppressWarnings(sapply(libs, require
                                            , character.only=TRUE
                                            , quietly=quietly
                                            , warn.conflicts=warn.conflicts
                                            ))
        if(any(!is.loaded)){
            if(install.missing){
                need.to.install <- libs[!is.loaded]
                message("Installing requested packages:", paste(need.to.install, collapse= ", "))
                install.packages( need.to.install, type=if(.Platform$OS.type=="unix") 'source' else 'both'
                                , quiet=quietly, verbose=verbose
                                )
                is.loaded <- sapply(libs, require, quietly=F, character.only=T)

            } else {
                stop("The following package(s) are not installed: ", paste(libs[!is.loaded], collapse=", "))
            }
        }
        invisible(is.loaded)
    }
}
unload <- 
function( ...
        , deparse=TRUE      #< deparse `...` to character strings or evaluate as regular R objects.
        , perl=FALSE        #< see <grep>
        , ignore.case=FALSE #< see <grep>
        , fixed=FALSE       #< see <grep>
        ){
    #! <detach> by regular expression.
    libs <- if(deparse) as.character(substitute(c(...)))[-1] else c(..., recursive=TRUE)
    if(length(libs)>1) 
        return(sapply( libs, unload, deparse=FALSE
                     , perl=perl, ignore.case=ignore.case, fixed=fixed
                     ))
    if(length(libs)==0) return(0)
    pos <- grep(libs, x=search(), perl=perl, ignore.case=ignore.case, fixed=fixed)
    if(length(pos) == 1){
        return(attr(detach(pos=pos), 'name'))
    }
    else if(length(pos) > 1){
        libs <- search()[pos]
        return(sapply(sapply(libs, detach), attr, 'name'))
        return(sapply(libs, unload, deparse=FALSE, fixed=TRUE))
    } else return(FALSE)
}
