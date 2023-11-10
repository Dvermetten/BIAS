
for (x in c("zoo", "lgarch", "Rcpp", "RcppArmadillo", 'nortest', 'data.table', 'goftest', 'ddst')){
    if (!require(x,character.only = TRUE))
    {
    install.packages(x,dep=TRUE)
        if(!require(x,character.only = TRUE)) stop("Package not found")
    }
}

if (!require("AutoSEARCH", character.only = TRUE)) {
    #install AutoSearch
    url <- "https://cran.r-project.org/src/contrib/Archive/AutoSEARCH/AutoSEARCH_1.5.tar.gz"
    pkgFile <- "AutoSEARCH_1.5.tar.gz"
    download.file(url = url, destfile = pkgFile)

    # Install package
    install.packages(pkgs=pkgFile, type="source", repos=NULL)

    # Delete package tarball
    unlink(pkgFile)
}

if (!require("PoweR", character.only = TRUE)) {
    #install PoweR
    url <- "https://cran.r-project.org/src/contrib/Archive/PoweR/PoweR_1.0.7.tar.gz"
    pkgFile <- "PoweR_1.0.7.tar.gz"
    download.file(url = url, destfile = pkgFile)

    # Install package
    install.packages(pkgs=pkgFile, type="source", repos=NULL)

    # Delete package tarball
    unlink(pkgFile)
}
