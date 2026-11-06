# Conditional descriptive associations.
run_associations <- function(project_root=getwd(),config_path=file.path(project_root,'config.json')) {
  packages <- c('fixest','arrow','data.table','jsonlite')
  absent <- packages[!vapply(packages, requireNamespace, logical(1), quietly=TRUE)]
  if(length(absent)) stop('Install R packages first: ', paste(absent,collapse=', '))
  cfg <- jsonlite::fromJSON(config_path)
  out <- cfg$output_dir
  absolute <- grepl('^[A-Za-z]:',out) || startsWith(out,'/') || startsWith(out,intToUtf8(92))
  if(!absolute) out <- file.path(project_root,out)
  dest <- file.path(out,'r');dir.create(dest,recursive=TRUE,showWarnings=FALSE)
  Sys.setenv(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='8')
  fixest::setFixest_nthreads(8)
  data.table::setDTthreads(4)
  p <- data.table::as.data.table(arrow::read_parquet(file.path(out,'data','r_papers.parquet')))
  scope <- jsonlite::fromJSON(file.path(out,'status','sample_scope.json'))
  if(scope$contract_version!=1 || scope$scope_rule!='observed_ai_score__source_venues__1980_2024' || !'scope_id' %in% names(p) || any(p$scope_id!=scope$scope_id))
    stop('R input belongs to another analysis. Rerun Python notebooks 00–04.')
  if(any(!is.finite(p$AIProb) | p$AIProb<0 | p$AIProb>1) ||
     any(p$year<cfg$ai_start | p$year>cfg$end_year) || anyDuplicated(p$paper_id))
    stop('R input needs unique papers, valid observed scores and publication years.')
  # Bit64 identifiers are converted directly to strings, never through R doubles.
  p[, source_id := as.character(source_id)]
  if(anyNA(p$source_id) || any(p$source_id=='')) stop('Missing publishing source IDs.')
  p[, year := as.integer(year)]
  p[, ai_gt05 := as.integer(AIProb>0.5)]
  p[, log_refs := log1p(n_references)]
  p[, log_team := log1p(team_size)]
  p[, log_c5 := log1p(c5_norm)]
  outcomes <- c('ref_field_simpson','emb_ref_pair_mean_distance',
                'emb_field_history_distance','mean_ref_age','log_c5')
  models <- list();rows <- list();coverage <- list()
  estimate <- function(data,outcome,score,label) {
    vars <- c(outcome,score,'log_refs','log_team','is_conference','year','source_id','sampling_weight')
    good <- complete.cases(data[, ..vars])
    if(outcome=='log_c5') good <- good & data$year<=cfg$end_year-5
    dd <- data[good]
    if(nrow(dd)<100 || length(unique(dd[[score]]))<2) return(NULL)
    sources <- data.table::uniqueN(dd$source_id)
    vc <- if(sources>=20) ~source_id else 'hetero'
    formula <- as.formula(paste(outcome,'~',score,'+ log_refs + log_team + is_conference | source_id^year'))
    fit <- fixest::feols(formula,data=dd,weights=~sampling_weight,vcov=vc,
                         lean=TRUE,mem.clean=TRUE,notes=FALSE)
    ct <- as.data.frame(fixest::coeftable(fit))
    co <- data.frame(model=label,outcome=outcome,term=rownames(ct),estimate=ct[[1]],
                     std_error=ct[[2]],statistic=ct[[3]],p_value=ct[[4]],row.names=NULL)
    ci <- as.data.frame(confint(fit))
    co$conf_low <- ci[match(co$term,rownames(ci)),1]
    co$conf_high <- ci[match(co$term,rownames(ci)),2]
    list(fit=fit,coefficients=co,coverage=data.frame(model=label,available_rows=nrow(data),
         complete_rows=nrow(dd),n_sources=sources,vcov=if(sources>=20)'source cluster' else 'heteroskedastic',
         score=score,outcome=outcome,sampling_fraction=cfg$r_export_fraction))
  }
  for(y in outcomes) for(score in c('AIProb','ai_gt05')) {
    label <- paste(y,score,sep='__')
    ans <- estimate(p,y,score,label)
    if(!is.null(ans)) {models[[label]] <- ans$fit;rows[[label]] <- ans$coefficients;coverage[[label]] <- ans$coverage}
    gc(FALSE)
  }
  # High-confidence restriction is a measurement sensitivity, not a treatment definition.
  pc <- p[AIProb<=0.2 | AIProb>0.8]
  ans <- estimate(pc,'ref_field_simpson','ai_gt05','ref_field_simpson__high_confidence')
  if(!is.null(ans)) {models[['high_confidence']]<-ans$fit;rows[['high_confidence']]<-ans$coefficients;coverage[['high_confidence']]<-ans$coverage}
  if(length(models)) {
    data.table::fwrite(data.table::rbindlist(rows,fill=TRUE),file.path(dest,'paper_association_coefficients.csv'))
    data.table::fwrite(data.table::rbindlist(coverage,fill=TRUE),file.path(dest,'paper_model_scope.csv'))
    fixest::etable(models,tex=TRUE,file=file.path(dest,'paper_associations.tex'),replace=TRUE)
    writeLines(capture.output(fixest::etable(models)),file.path(dest,'paper_associations.txt'))
    saveRDS(models,file.path(dest,'paper_associations.rds'))
    co <- data.table::rbindlist(rows,fill=TRUE)
    grDevices::pdf(file.path(dest,'conditional_association_coefficients.pdf'),width=7.08,height=7.8,pointsize=8)
    graphics::par(mfrow=c(3,2),mar=c(4,5,3,1),cex=0.8)
    for(y in outcomes) {
      zz <- co[outcome==y & term %in% c('AIProb','ai_gt05') & !grepl('confidence',model)]
      if(!nrow(zz)) {graphics::plot.new();next}
      limits <- range(c(0,zz$conf_low,zz$conf_high),finite=TRUE)
      if(diff(limits)==0) limits <- limits+c(-0.01,0.01)
      graphics::plot(zz$estimate,seq_len(nrow(zz)),xlim=limits,ylim=c(0.5,nrow(zz)+0.5),
         yaxt='n',ylab='',xlab='Conditional association',main=paste(strwrap(y,width=24),collapse='\n'),pch=19)
      graphics::axis(2,at=seq_len(nrow(zz)),labels=zz$term,las=1)
      graphics::segments(zz$conf_low,seq_len(nrow(zz)),zz$conf_high,seq_len(nrow(zz)))
      graphics::abline(v=0,lty=3,col='grey50')
    }
    graphics::plot.new();graphics::text(0.5,0.5,'95% model intervals\nSeparate outcome scales\nDescriptive associations',cex=0.8)
    grDevices::dev.off()
  }
  # Country-source cells, weighted by unique-paper fractional country credit.
  cf <- data.table::fread(file.path(out,'tables','country_source_rca_annual.csv'))
  if(!'scope_id' %in% names(cf) || any(cf$scope_id!=scope$scope_id)) stop('Country table is from another sample.')
  cf <- cf[year>=cfg$trend_start & known>=cfg$min_cell_period_mass]
  cf[,country:=as.factor(country)];cf[,source_id:=as.factor(as.character(source_id))]
  if(nrow(cf)>=100 && data.table::uniqueN(cf$country)>=5) {
    fit <- fixest::feols(ai_share ~ asinh(rca_known) | country + source_id^year,
          data=cf,weights=~known,vcov=~country,lean=TRUE,mem.clean=TRUE)
    fixest::etable(fit,tex=TRUE,file=file.path(dest,'country_source_associations.tex'),replace=TRUE)
    saveRDS(fit,file.path(dest,'country_source_associations.rds'))
    data.table::fwrite(data.frame(term=rownames(fixest::coeftable(fit)),as.data.frame(fixest::coeftable(fit)),check.names=FALSE),
                      file.path(dest,'country_source_coefficients.csv'))
  }
  writeLines(c('These are conditional descriptive associations.',
     'AI-positive: AIProb>0.5; Other-scored: observed AIProb<=0.5.',
     'One row per scored paper, with publishing-source by year fixed effects.',
     'Uncertainty is clustered by source where at least 20 sources are available.',
     'C5 outcomes use only mature cohorts through 2019; no C5 result is imputed for 2020-2024.',
     'Regression intervals condition on observed records and model; they do not account for database or classifier error.'),
     file.path(dest,'interpretation.txt'))
  writeLines(capture.output(sessionInfo()),file.path(dest,'R_session.txt'))
  invisible(models)
}
