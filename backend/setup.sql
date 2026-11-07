-- Run once in a new Supabase project's SQL Editor. Safe to rerun.
begin;

create table if not exists public.edge_records (
  id uuid primary key,
  owner_id uuid not null references auth.users(id),
  session_id uuid not null,
  kind text not null check (kind in ('SESSION','OBSERVATION','LOCATION_BATCH','TRACK_BATCH')),
  revision bigint not null check (revision >= 1),
  payload jsonb not null check (jsonb_typeof(payload) = 'object' and payload->>'schema_version' = '1'),
  first_received_at timestamptz not null default clock_timestamp(),
  updated_at timestamptz not null default clock_timestamp()
);
create index if not exists edge_records_session_idx on public.edge_records(owner_id,session_id,kind);
create index if not exists edge_records_recent_idx on public.edge_records(owner_id,updated_at desc);
alter table public.edge_records enable row level security;
revoke all on public.edge_records from anon;
grant select, insert, update on public.edge_records to authenticated;
drop policy if exists edge_own_select on public.edge_records;
create policy edge_own_select on public.edge_records for select to authenticated using ((select auth.uid()) = owner_id);
drop policy if exists edge_own_insert on public.edge_records;
create policy edge_own_insert on public.edge_records for insert to authenticated with check ((select auth.uid()) = owner_id);
drop policy if exists edge_own_update on public.edge_records;
create policy edge_own_update on public.edge_records for update to authenticated
  using ((select auth.uid()) = owner_id) with check ((select auth.uid()) = owner_id);

-- Supabase exposes this as POST /rest/v1/rpc/ingest_edge_records.
-- Acks name the offered revision; Android acknowledges it only if it is still current locally.
-- Newer server revisions cannot be overwritten by retries of older snapshots.
create or replace function public.ingest_edge_records(p_records jsonb)
returns jsonb language plpgsql security invoker set search_path = '' as $$
declare
  entry jsonb;
  record_uuid uuid;
  session_uuid uuid;
  offered_revision bigint;
  offered_kind text;
  value jsonb;
  accepted jsonb := '[]'::jsonb;
  rejected jsonb := '[]'::jsonb;
  current_owner uuid := auth.uid();
begin
  if current_owner is null then raise exception 'Authentication required' using errcode='28000'; end if;
  if jsonb_typeof(p_records) is distinct from 'array' then
    raise exception 'p_records must be an array' using errcode='22023';
  end if;
  if jsonb_array_length(p_records)>100 then raise exception 'Maximum batch size is 100' using errcode='22023'; end if;
  for entry in select * from jsonb_array_elements(p_records) loop
    begin
      record_uuid := (entry->>'id')::uuid;
      session_uuid := (entry->>'session_id')::uuid;
      offered_revision := (entry->>'revision')::bigint;
      offered_kind := entry->>'kind';
      value := entry->'payload';
      if record_uuid is null or session_uuid is null or offered_revision is null or offered_revision<1
        or offered_kind is null or offered_kind not in ('SESSION','OBSERVATION','LOCATION_BATCH','TRACK_BATCH')
        or jsonb_typeof(value) is distinct from 'object' or value->>'schema_version' is distinct from '1'
        or value->>'session_id' is distinct from session_uuid::text or pg_column_size(value)>1048576 then
        raise exception 'Invalid record envelope or payload' using errcode='22023';
      end if;
      if offered_kind='SESSION' and record_uuid<>session_uuid then
        raise exception 'Session ID mismatch' using errcode='22023';
      end if;
      if offered_kind='OBSERVATION' and value->>'observation_id' is distinct from record_uuid::text then
        raise exception 'Observation ID mismatch' using errcode='22023';
      end if;
      -- The application uploads an immutable crop before publishing its reference.
      if value->'evidence'->>'storage_path' is not null and (
        split_part(value->'evidence'->>'storage_path','/',1) <> current_owner::text or
        split_part(value->'evidence'->>'storage_path','/',2) <> session_uuid::text) then
        raise exception 'Evidence path does not belong to this owner/session' using errcode='22023';
      end if;
      if value->'scene_context_image'->>'storage_path' is not null and (
        split_part(value->'scene_context_image'->>'storage_path','/',1) <> current_owner::text or
        split_part(value->'scene_context_image'->>'storage_path','/',2) <> session_uuid::text) then
        raise exception 'Scene path does not belong to this owner/session' using errcode='22023';
      end if;
      insert into public.edge_records as stored(id,owner_id,session_id,kind,revision,payload)
        values(record_uuid,current_owner,session_uuid,offered_kind,offered_revision,value)
      on conflict(id) do update set revision=excluded.revision,payload=excluded.payload,updated_at=clock_timestamp()
        where stored.owner_id=current_owner and stored.session_id=excluded.session_id
          and stored.kind=excluded.kind and stored.revision<excluded.revision;
      -- Conflicting identities must not be silently acknowledged.
      if not exists(select 1 from public.edge_records r where r.id=record_uuid and r.owner_id=current_owner
        and r.session_id=session_uuid and r.kind=offered_kind and r.revision>=offered_revision) then
        raise exception 'Record identity conflict' using errcode='22023';
      end if;
      accepted := accepted || jsonb_build_array(jsonb_build_object('id',record_uuid,'revision',offered_revision));
    exception when others then
      rejected := rejected || jsonb_build_array(jsonb_build_object('id',coalesce(entry->>'id',''),
        'revision',coalesce(entry->>'revision','0'),'reason',sqlerrm));
    end;
  end loop;
  return jsonb_build_object('accepted',accepted,'rejected',rejected);
end;
$$;
revoke all on function public.ingest_edge_records(jsonb) from public,anon;
grant execute on function public.ingest_edge_records(jsonb) to authenticated;

-- Readable projections. security_invoker keeps the base table's owner policy in force.
drop view if exists public.survey_sessions;
drop view if exists public.lamp_observations;
create or replace view public.survey_sessions with (security_invoker=true) as
select id,owner_id,revision,payload->>'state' as state,
  to_timestamp((payload->>'start_utc_ms')::double precision/1000) as started_at,
  (payload->>'tracks')::integer as tracks,
  payload->'diagnostics' as diagnostics,payload,updated_at
from public.edge_records where kind='SESSION';

create or replace view public.lamp_observations with (security_invoker=true) as
select id,owner_id,session_id,revision,(payload->>'track_id')::bigint as track_id,
  (payload->>'finished')::boolean as finished,
  (payload->'device_observation_location'->>'latitude')::double precision as device_latitude,
  (payload->'device_observation_location'->>'longitude')::double precision as device_longitude,
  (payload->'device_observation_location'->>'accuracy_m')::double precision as accuracy_m,
  payload->'evidence'->>'storage_path' as crop_path,payload,updated_at
from public.edge_records where kind='OBSERVATION';

create or replace view public.survey_location_samples with (security_invoker=true) as
select r.owner_id,r.session_id,r.id as batch_id,s.value as sample,r.updated_at
from public.edge_records r cross join lateral jsonb_array_elements(r.payload->'locations') s(value)
where r.kind='LOCATION_BATCH';

create or replace view public.survey_track_samples with (security_invoker=true) as
select r.owner_id,r.session_id,r.id as batch_id,s.value->>'observation_id' as observation_id,s.value as sample,r.updated_at
from public.edge_records r cross join lateral jsonb_array_elements(r.payload->'samples') s(value)
where r.kind='TRACK_BATCH';

grant select on public.survey_sessions,public.lamp_observations,public.survey_location_samples,public.survey_track_samples to authenticated;
revoke all on public.survey_sessions,public.lamp_observations,public.survey_location_samples,public.survey_track_samples from anon;

insert into storage.buckets(id,name,public,file_size_limit,allowed_mime_types)
values('evidence','evidence',false,67108864,array['image/jpeg','image/png','application/gzip','application/json'])
on conflict(id) do update set public=false,file_size_limit=excluded.file_size_limit,allowed_mime_types=excluded.allowed_mime_types;
drop policy if exists evidence_own_select on storage.objects;
create policy evidence_own_select on storage.objects for select to authenticated
  using(bucket_id='evidence' and (storage.foldername(name))[1]=(select auth.uid())::text);
drop policy if exists evidence_own_insert on storage.objects;
create policy evidence_own_insert on storage.objects for insert to authenticated
  with check(bucket_id='evidence' and (storage.foldername(name))[1]=(select auth.uid())::text);
drop policy if exists evidence_own_update on storage.objects;
create policy evidence_own_update on storage.objects for update to authenticated
  using(bucket_id='evidence' and (storage.foldername(name))[1]=(select auth.uid())::text)
  with check(bucket_id='evidence' and (storage.foldername(name))[1]=(select auth.uid())::text);

do $$ begin
  if exists(select 1 from pg_publication where pubname='supabase_realtime') and not exists(
    select 1 from pg_publication_tables where pubname='supabase_realtime' and schemaname='public' and tablename='edge_records') then
    alter publication supabase_realtime add table public.edge_records;
  end if;
end $$;
commit;
