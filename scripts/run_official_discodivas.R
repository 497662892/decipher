# Preserve the upstream script while making its positional column indexing valid.
# In its regress.PCA=FALSE branch, data.table's dat[, 1+i] returns a constant,
# whereas the upstream weighted sum expects data.frame column-selection semantics.
options(datatable.fread.datatable = FALSE)
upstream_path <- Sys.getenv("DECIPEHR_DISCODIVAS_SOURCE")
if (!nzchar(upstream_path)) stop("DECIPEHR_DISCODIVAS_SOURCE is required")
source(upstream_path)
