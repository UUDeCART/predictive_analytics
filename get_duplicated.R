get_duplicated_<- function(.data, key.col, ..., .dots){
    .names <- names(collect(head(.data,1)))
    dots <- lazyeval::all_dots(.dots, ...)
    vars <- select_vars_(.names, dots)
    key.col <- select_vars_(vars=.names, args=key.col)
    all.vars <- c(key.col, dots)
    .data %>% 
        select_(.dots=all.vars) %>% distinct %>%
        count_(key.col) %>% 
        filter_(~n>1) %>% select(-n) %>%
        left_join(.data %>% select_(.dots=all.vars) %>% distinct
                 , by=key.col)
}
get_duplicated <- function(.data, key, ...){
    key.col <- tidyr:::col_name(substitute(key))
    get_duplicated_(.data, key.col=key.col, .dots = lazyeval::lazy_dots(...))
}